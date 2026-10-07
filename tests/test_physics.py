"""Unit tests for the Hamiltonian, channels, reservoir dynamics and diagnostics
(Methods 2.10, checks 1-3)."""
import numpy as np
import pytest
from scipy.linalg import expm

from qrc_qmi import diagnostics as dg
from qrc_qmi.channels import (LocalChannel, apply_local_kraus, choi_matrix, jump_operator,
                              kraus_operators, superoperator)
from qrc_qmi.hamiltonian import SZ, bit_table, build_hamiltonian, draw_disorder, embed, gap_ratio
from qrc_qmi.reservoir import (Reservoir, input_state, integrator_error, liouvillian,
                               ode_reference_error, trace_distance)

ALL = ["dephasing", "amplitude_damping", "phase_flip", "bit_flip"]


def random_rho(d, rng, B=1):
    A = rng.normal(size=(B, d, d)) + 1j * rng.normal(size=(B, d, d))
    rho = A @ A.conj().transpose(0, 2, 1)
    return rho / np.trace(rho, axis1=1, axis2=2)[:, None, None]


def make_H(N, h=1.0, W=0.5, seed=0):
    J, D = draw_disorder(N, W, np.random.default_rng(seed))
    return build_hamiltonian(N, h, J, D)


# ----------------------------------------------------------------------------- Hamiltonian
def test_hamiltonian_symmetric_and_parity_conserving():
    N = 5
    H = make_H(N)
    assert np.allclose(H, H.T)
    P = np.diag(np.prod(1.0 - 2.0 * bit_table(N), axis=1))
    assert np.allclose(H @ P, P @ H)


def test_hamiltonian_matches_explicit_two_qubit_form():
    J = np.array([[0, 0.3], [0, 0]]); D = np.array([0.1, -0.2]); h = 0.7
    X = np.array([[0, 1], [1, 0]])
    ref = 0.3 * np.kron(X, X) + 0.5 * ((h + 0.1) * np.kron(SZ, np.eye(2)) + (h - 0.2) * np.kron(np.eye(2), SZ))
    assert np.allclose(build_hamiltonian(2, h, J, D), ref)


def test_gap_ratio_limits():
    """Poisson value for a strongly disordered (localised) chain, GOE-like for weak disorder."""
    r_loc = np.mean([gap_ratio(make_H(8, h=1.0, W=20.0, seed=s), 8) for s in range(20)])
    r_erg = np.mean([gap_ratio(make_H(8, h=1.0, W=0.1, seed=s), 8) for s in range(20)])
    assert abs(r_loc - 0.386) < 0.03
    assert r_erg > 0.48


# ----------------------------------------------------------------------------- channels
@pytest.mark.parametrize("name", ALL)
def test_kraus_is_exact_lindblad_solution(name):
    gamma, t = 0.37, 1.3
    K = kraus_operators(name, gamma, t)
    assert np.allclose(sum(k.conj().T @ k for k in K), np.eye(2))          # trace preserving
    S_lindblad = expm(liouvillian(np.zeros((2, 2)), [jump_operator(name, gamma)]) * t)
    assert np.allclose(superoperator(K), S_lindblad, atol=1e-12)


@pytest.mark.parametrize("name", ALL)
def test_choi_positive_and_trace_preserving(name):
    J = choi_matrix(kraus_operators(name, 0.8, 0.9))
    assert np.linalg.eigvalsh(J).min() > -1e-12
    # partial trace over the output factor gives the identity on the input
    assert np.allclose(np.einsum("iaja->ij", J.reshape(2, 2, 2, 2)), np.eye(2))


def test_equal_coherence_decay_and_fixed_points():
    gamma, t = 0.5, 2.0
    plus = np.full((2, 2), 0.5, dtype=complex)
    for name in ("dephasing", "amplitude_damping"):
        out = sum(k @ plus @ k.conj().T for k in kraus_operators(name, gamma, t))
        assert np.isclose(abs(out[0, 1]), 0.5 * np.exp(-gamma * t / 2))    # T2 = 2/gamma
    one = np.diag([0.0, 1.0]).astype(complex)
    out = sum(k @ one @ k.conj().T for k in kraus_operators("amplitude_damping", 1.0, 50.0))
    assert np.allclose(out, np.diag([1.0, 0.0]), atol=1e-10)               # relaxes to |0>
    assert np.allclose(superoperator(kraus_operators("phase_flip", 0.1, 1.0)),
                       superoperator(kraus_operators("dephasing", 0.4, 1.0)))


@pytest.mark.parametrize("name", ALL)
def test_fast_channel_matches_reference(name):
    N, rng = 4, np.random.default_rng(1)
    rho = random_rho(2**N, rng, B=3)
    ch = LocalChannel(name, 0.6, 0.7, N)
    ref = apply_local_kraus(rho.copy(), ch.kraus, N)
    out = ch(rho.copy())
    assert np.allclose(out, ref, atol=1e-12)
    assert np.allclose(np.trace(out, axis1=1, axis2=2), 1.0)
    # explicit full-space Kraus check on one qubit
    K = [embed(k, 0, N) for k in ch.kraus]
    one = sum(k @ rho[0] @ k.conj().T for k in K)
    for i in range(1, N):
        one = sum(embed(k, i, N) @ one @ embed(k, i, N).conj().T for k in ch.kraus)
    assert np.allclose(out[0], one, atol=1e-12)


# ----------------------------------------------------------------------------- reservoir
def test_injection_resets_qubit_zero():
    N, rng = 3, np.random.default_rng(2)
    res = Reservoir(H=make_H(N), N=N)
    rho = random_rho(8, rng, B=2)
    s = np.array([0.3, 0.8])
    out = res.inject(rho, s)
    for b in range(2):
        assert np.allclose(dg.partial_trace(out[b], [0], N), input_state(s[b:b + 1])[0])
        assert np.allclose(dg.partial_trace(out[b], [1, 2], N), dg.partial_trace(rho[b], [1, 2], N))
        assert np.allclose(out[b], np.kron(input_state(s[b:b + 1])[0], dg.partial_trace(rho[b], [1, 2], N)))


@pytest.mark.parametrize("enc", ["pure_z", "mixed_z", "pure_x"])
def test_input_states_are_valid(enc):
    r = input_state(np.linspace(0, 1, 7), enc)
    assert np.allclose(np.trace(r, axis1=1, axis2=2), 1.0)
    assert np.allclose(r, r.conj().transpose(0, 2, 1))
    assert np.linalg.eigvalsh(r).min() > -1e-12
    if enc != "mixed_z":
        assert np.allclose(np.einsum("bij,bji->b", r, r).real, 1.0)        # pure


@pytest.mark.parametrize("name", ALL)
def test_strang_converges_to_exact_at_second_order(name):
    N = 3
    H = make_H(N, seed=3)
    e = [integrator_error(H, N, name, 0.3, n_s, periods=5) for n_s in (4, 8, 16)]
    assert e[2] < 1e-4
    assert 3.0 < e[0] / e[1] < 5.0 and 3.0 < e[1] / e[2] < 5.0             # error ~ n_s^-2


@pytest.mark.parametrize("name", ["dephasing", "amplitude_damping"])
@pytest.mark.parametrize("prop", ["exact", "strang"])
def test_propagators_match_independent_ode_solver(name, prop):
    N = 3
    res = Reservoir(H=make_H(N, seed=4), N=N, channel=name, gamma=0.2, propagator=prop, n_s=64)
    assert ode_reference_error(res, periods=3) < (1e-7 if prop == "exact" else 1e-4)


def test_states_stay_physical_and_features_are_expectation_values():
    N = 4
    res = Reservoir(H=make_H(N, seed=5), N=N, channel="amplitude_damping", gamma=0.1,
                    propagator="strang", n_s=8)
    s = np.random.default_rng(0).uniform(size=(2, 20))
    feats, rho = res.run(s)
    assert feats.shape == (2, 20, res.V * (N + N * (N - 1) // 2))
    for b in range(2):
        assert np.isclose(np.trace(rho[b]).real, 1.0)
        assert np.allclose(rho[b], rho[b].conj().T)
        assert np.linalg.eigvalsh(rho[b]).min() > -1e-10
    Z = [embed(SZ, i, N) for i in range(N)]
    last = feats[0, -1].reshape(res.V, -1)[-1]
    assert np.isclose(last[0], np.trace(rho[0] @ Z[0]).real)
    assert np.isclose(last[N], np.trace(rho[0] @ Z[0] @ Z[1]).real)       # first pair is (0, 1)
    assert np.isclose(last[-1], np.trace(rho[0] @ Z[N - 2] @ Z[N - 1]).real)


def test_echo_state_property():
    N = 4
    res = Reservoir(H=make_H(N, seed=6), N=N, channel="dephasing", gamma=0.05)
    s = np.random.default_rng(0).uniform(size=300)
    assert res.echo_state_distance(s, np.random.default_rng(1)) < 1e-6


# ----------------------------------------------------------------------------- diagnostics
def ket(v):
    v = np.asarray(v, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def test_bell_pair():
    rho = ket([1, 0, 0, 1])
    assert np.isclose(dg.mean_negativity(rho, 2), 0.5)
    assert np.isclose(dg.total_mutual_information(rho, 2), 2.0)
    assert np.isclose(dg.mean_negativity(rho, 2, log=True), 1.0)
    assert np.isclose(dg.classical_correlations(rho, 2), 1.0)


@pytest.mark.parametrize("N", [2, 3, 5])
def test_plus_state(N):
    rho = ket(np.ones(2**N))
    assert np.isclose(dg.l1_coherence(rho, normalised=False), 2**N - 1)
    assert np.isclose(dg.l1_coherence(rho), 1.0)
    assert abs(dg.total_mutual_information(rho, N)) < 1e-9
    assert dg.mean_negativity(rho, N) < 1e-12
    # hookup = T + local coherence = 0 + N bits
    assert np.isclose(dg.hookup(rho, N), N)


def test_product_and_classically_correlated_states_have_zero_negativity():
    rng = np.random.default_rng(0)
    a, b, c = (random_rho(2, rng)[0] for _ in range(3))
    prod = np.kron(np.kron(a, b), c)
    assert dg.mean_negativity(prod, 3) < 1e-12
    assert abs(dg.total_mutual_information(prod, 3)) < 1e-9
    cc = np.diag([0.5, 0, 0, 0.5]).astype(complex)                          # (|00><00| + |11><11|)/2
    assert dg.mean_negativity(cc, 2) < 1e-12
    assert np.isclose(dg.total_mutual_information(cc, 2), 1.0)
    assert np.isclose(dg.classical_correlations(cc, 2), 1.0)
    assert dg.l1_coherence(cc) == 0.0


def test_ghz_state():
    N = 4
    v = np.zeros(2**N); v[0] = v[-1] = 1
    rho = ket(v)
    assert np.isclose(dg.mean_negativity(rho, N), 0.5)                      # 1/2 for every cut
    assert np.isclose(dg.total_mutual_information(rho, N), N)               # N * 1 - 0
    assert np.isclose(dg.pairwise_mutual_information(rho, N), 1.0)          # classical bit per pair
    assert np.isclose(dg.half_chain_mutual_information(rho, N), 2.0)


def test_pure_state_mutual_information_is_twice_entanglement_entropy():
    N, rng = 4, np.random.default_rng(3)
    rho = ket(rng.normal(size=16) + 1j * rng.normal(size=16))
    SA = dg.entropy(dg.partial_trace(rho, [0, 1], N))
    assert np.isclose(dg.half_chain_mutual_information(rho, N), 2 * SA)


def test_partial_trace_against_kron():
    rng = np.random.default_rng(4)
    a, b, c = random_rho(2, rng)[0], random_rho(2, rng)[0], random_rho(2, rng)[0]
    rho = np.kron(np.kron(a, b), c)
    assert np.allclose(dg.partial_trace(rho, [1], 3), b)
    assert np.allclose(dg.partial_trace(rho, [0, 2], 3), np.kron(a, c))
    assert np.isclose(trace_distance(a, a), 0.0)


def test_all_diagnostics_consistent():
    N, rng = 3, np.random.default_rng(5)
    rho = random_rho(8, rng)[0]
    d = dg.all_diagnostics(rho, N)
    assert np.isclose(d["tmi"], dg.total_mutual_information(rho, N))
    assert np.isclose(d["pair_mi"], dg.pairwise_mutual_information(rho, N))
    assert np.isclose(d["negativity"], dg.mean_negativity(rho, N))
    assert d["hookup"] >= d["tmi"] - 1e-9 and d["tmi"] >= d["classical_corr"] - 1e-9
