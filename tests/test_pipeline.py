"""Tests for tasks, readout, IPC, the ESN baseline, the statistics and the sweep runner."""
import numpy as np
import pandas as pd
import pytest

from qrc_qmi import stats as st
from qrc_qmi.config import SMOKE
from qrc_qmi.esn import esn_states
from qrc_qmi.ipc import capacities, ipc, target_index, target_matrix
from qrc_qmi.readout import LAMBDAS, Splits, add_shot_noise, linear_quadratic_features, nrmse, ridge_select
from qrc_qmi.sweep import make_inputs, run_config
from qrc_qmi.tasks import mackey_glass, narma_inputs, narma_target


# ----------------------------------------------------------------------------- tasks
def test_narma10_recurrence():
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 0.5, 300)
    t = narma_target(u, 10)
    y = np.concatenate([[0.0], t])                     # y[k+1] = t[k]
    for k in (50, 123, 299):
        ref = 0.3 * y[k] + 0.05 * y[k] * y[k - 9:k + 1].sum() + 1.5 * u[k - 9] * u[k] + 0.1
        assert np.isclose(t[k], ref)
    # the target at step k must not depend on future inputs
    u2 = u.copy(); u2[200:] = rng.uniform(0, 0.5, 100)
    assert np.allclose(narma_target(u2, 10)[:200], t[:200])
    assert not np.isclose(narma_target(u2, 10)[200], t[200])


def test_narma_inputs_shapes_and_range():
    s, targets, disc = narma_inputs(400, 3, seed=1, tasks=("narma5", "narma10", "narma20"))
    assert s.shape == (3, 400) and s.min() >= 0 and s.max() <= 1
    assert all(np.all(np.isfinite(v)) for v in targets.values())
    assert np.abs(targets["narma20"]).max() < 1.0      # tanh-bounded
    s2, _, _ = narma_inputs(400, 3, seed=1, tasks=("narma5", "narma10", "narma20"))
    assert np.array_equal(s, s2)                       # deterministic


def test_mackey_glass_is_bounded_and_chaotic_looking():
    x = mackey_glass(3000, seed=0)
    assert 0.2 < x.min() < 0.6 and 1.2 < x.max() < 1.4
    assert 0.15 < x.std() < 0.3
    # converged with respect to the integration step
    kw = dict(seed=0, discard=0.0, history=1.2)
    assert np.abs(mackey_glass(150, dt_int=0.05, **kw) - mackey_glass(150, dt_int=0.1, **kw)).max() < 1e-4


# ----------------------------------------------------------------------------- readout
def test_ridge_matches_closed_form_and_recovers_linear_map():
    rng = np.random.default_rng(0)
    sp = Splits(washout=10, n_train=400, n_val=100, n_test=100)
    X = rng.normal(size=(sp.total, 8))
    w = rng.normal(size=8)
    y = X @ w + 0.3 + 0.01 * rng.normal(size=sp.total)
    fit = ridge_select(X, y, sp)
    assert fit["test_nrmse"][0] < 0.02
    lam = 1e-2
    f1 = ridge_select(X, y, sp, lambdas=[lam])
    Xc = X[sp.train] - X[sp.train].mean(0)
    wr = np.linalg.solve(Xc.T @ Xc + lam * np.eye(8), Xc.T @ (y[sp.train] - y[sp.train].mean()))
    pred = (X[sp.test] - X[sp.train].mean(0)) @ wr + y[sp.train].mean()
    assert np.isclose(f1["test_nrmse"][0], nrmse(y[sp.test], pred))
    assert len(LAMBDAS) == 21 and np.isclose(LAMBDAS[0], 1e-10) and np.isclose(LAMBDAS[-1], 1.0)


def test_shot_noise_variance():
    rng = np.random.default_rng(0)
    X = np.full((200000, 1), 0.6)
    Y = add_shot_noise(X, 1e4, rng)
    assert np.isclose(Y.std(), np.sqrt((1 - 0.36) / 1e4), rtol=0.02)
    assert np.isclose(add_shot_noise(X, None, rng, "gaussian", sigma=1e-3).std(), 1e-3, rtol=0.02)
    assert add_shot_noise(np.array([[1.0, -1.0]]), 10, rng).tolist() == [[1.0, -1.0]]
    assert add_shot_noise(X, None, rng) is X


def test_linear_quadratic_features():
    s = np.arange(1.0, 21.0)
    F = linear_quadratic_features(s, 3)
    assert F.shape == (20, 3 + 6)
    assert F[10, :3].tolist() == [11.0, 10.0, 9.0] and F[10, 3] == 121.0 and F[0, 1] == 0.0


# ----------------------------------------------------------------------------- IPC
def test_ipc_target_count_and_orthonormality():
    idx = target_index(3, 20)
    assert len(idx) == 21 + (21 + 210) + (21 + 420 + 1330)
    u = np.random.default_rng(0).uniform(-1, 1, 200000)
    Z = target_matrix(u, target_index(3, 2), 3, 2)
    G = Z.T @ Z / len(Z)
    assert np.abs(G - np.eye(G.shape[0])).max() < 0.03


def test_ipc_of_shift_register_and_quadratic_reservoir():
    rng = np.random.default_rng(1)
    u = rng.uniform(-1, 1, 6000)
    lag = lambda k: np.concatenate([np.zeros(k), u[:len(u) - k]])
    X = np.stack([lag(k) for k in range(4)], axis=1)            # remembers 4 inputs linearly
    out = ipc(X, u, max_degree=2, max_delay=6)
    assert np.isclose(out["ipc_deg1"], 4.0, atol=0.02) and out["ipc_deg2"] < 0.02   # finite-T leakage
    X2 = np.stack([lag(0), lag(1), lag(0) * lag(1), lag(2) ** 2], axis=1)
    out = ipc(X2, u, max_degree=2, max_delay=6)
    assert np.isclose(out["ipc_total"], 4.0, atol=0.02)         # total capacity = rank
    assert np.isclose(out["ipc_deg1"], 2.0, atol=0.02) and np.isclose(out["ipc_deg2"], 2.0, atol=0.02)
    assert capacities(rng.normal(size=(5994, 4)), target_matrix(u, target_index(1, 6), 1, 6)).max() < 0.01


# ----------------------------------------------------------------------------- ESN
def test_esn_spectral_radius_shapes_and_echo_state():
    s = np.random.default_rng(0).uniform(size=(2, 300))
    X = esn_states(s, 30, leak=0.5, radius=0.9, in_scale=1.0, bias_scale=0.2, seed=1)
    assert X.shape == (2, 300, 30) and np.abs(X).max() <= 1.0
    assert np.array_equal(X, esn_states(s, 30, leak=0.5, radius=0.9, in_scale=1.0, bias_scale=0.2, seed=1))
    s2 = s.copy(); s2[:, :50] = 0.0                              # different early history
    X2 = esn_states(s2, 30, leak=0.5, radius=0.9, in_scale=1.0, bias_scale=0.2, seed=1)
    assert np.abs(X[:, -1] - X2[:, -1]).max() < 1e-8


# ----------------------------------------------------------------------------- statistics
def synthetic(n_cells=120, per=4, beta_t=0.0, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    n = n_cells * per
    cell = np.repeat(np.arange(n_cells), per)
    ce = np.repeat(rng.normal(size=n_cells), per)               # shared within a cell
    neg, coh = rng.normal(size=n) + ce, rng.normal(size=n) + 0.5 * ce
    tmi = 0.6 * coh + 0.8 * rng.normal(size=n)
    ch = np.repeat(rng.integers(0, 2, n_cells), per).astype(float)
    y = 0.5 * neg + 0.7 * coh + beta_t * tmi + shift + 0.5 * rng.normal(size=n) + 0.2 * ce
    return pd.DataFrame({st.NEG: neg, st.COH: coh, st.TMI: tmi, st.PAIR: tmi + 0.1 * rng.normal(size=n),
                         "channel_ad": ch, "channel": np.where(ch == 1, "amplitude_damping", "dephasing"),
                         "cell": [f"c{c}" for c in cell], "delta": y})


def test_h1a_detects_real_gain_and_declares_redundancy():
    kw = dict(controls=["channel_ad"], n_boot=150, n_repeats=3, seed=1)
    yes = st.nested_gain(synthetic(beta_t=0.6), **kw)
    assert yes["decision"] == "supported" and yes["gain"] > 0.05
    no = st.nested_gain(synthetic(beta_t=0.0, n_cells=300), **kw)
    assert no["decision"] == "redundant" and abs(no["gain"]) < 0.01


def test_cv_groups_never_straddle_folds():
    df = synthetic()
    g = df["cell"].to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    fold = (np.random.default_rng(0).permutation(len(uniq)) % 5)[inv]
    assert all(len(set(fold[g == c])) == 1 for c in uniq)
    i = st._cluster_resample(g, np.random.default_rng(0))
    assert len(i) == len(df) and all((np.sum(g[i] == c) % 4) == 0 for c in uniq)


def test_commonality_components_sum_to_total():
    c = st.commonality(synthetic(beta_t=0.4), controls=["channel_ad"]).set_index("component")["r2"]
    parts = c.drop(["controls only (R2)", "all three (increment)"]).sum()
    assert np.isclose(parts, c["all three (increment)"])


def test_h1b_interaction_and_tost():
    df = synthetic(beta_t=0.5, n_cells=400)
    same = st.h1b(df, controls=["channel_ad"])
    assert (same["decision"] == "channel-independent").all()
    df2 = df.copy()
    df2["delta"] += 0.8 * df2["channel_ad"] * df2[st.TMI]
    diff = st.h1b(df2, controls=["channel_ad"])
    assert diff.loc[0, "decision"] == "channel-dependent" and diff.loc[0, "interaction"] > 0.2
    assert np.allclose(st.holm([0.01, 0.04]), [0.02, 0.04]) and np.allclose(st.holm([0.04, 0.03]), [0.06, 0.06])


def test_transfer_ratio():
    a, b = synthetic(beta_t=0.5, seed=1), synthetic(beta_t=0.5, seed=2)
    same = st.transfer(a, b, cols=(st.NEG, st.COH, st.TMI), n_boot=100)
    assert 0.85 < same["ratio"] < 1.15 and same["passes"]
    c = synthetic(beta_t=-0.5, seed=3)
    c["delta"] = -c["delta"]
    bad = st.transfer(a, c, cols=(st.NEG, st.COH, st.TMI), n_boot=100)
    assert bad["ratio"] < 0 and not bad["passes"]


# ----------------------------------------------------------------------------- sweep
def test_run_config_smoke():
    inputs = make_inputs(SMOKE)
    pts = list(SMOKE.grid())
    row = run_config(SMOKE, pts[0], inputs, {})
    again = run_config(SMOKE, pts[0], inputs, {})
    for k in ("nrmse_narma10_s6", "nrmse_narma5_s4", "nrmse_mackey_glass_s6", "diag_tmi",
              "diagmg_coherence", "ipc_total", "gap_ratio", "esp_dist"):
        assert np.isfinite(row[k]), k
        assert row[k] == again[k], k                              # bit-for-bit reproducible
    assert row["D"] == SMOKE.n_features(3) and row["ipc_total"] <= row["D"]
    assert row["nrmse_narma10_s6"] <= row["nrmse_narma10_s4"] + 0.05
    other = run_config(SMOKE, pts[1], inputs, {})
    assert other["diag_tmi"] != row["diag_tmi"]                   # independent Hamiltonian draw
