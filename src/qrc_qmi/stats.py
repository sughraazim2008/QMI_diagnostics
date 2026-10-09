"""Statistical analysis (Methods 2.9, design v2): nested cross-validated models with a
cluster bootstrap (H1a), channel interaction with cluster-robust errors (H1b), transfer (H1c).

Clusters are *sites*: one (N, gamma, h, W) grid point.  Both channels at a site use the same
Hamiltonian realisations, so a site is the unit that is independent of the others.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

NEG, COH, TMI, PAIR = "diag_negativity", "diag_coherence", "diag_tmi_per_qubit", "diag_pair_mi"
LOGNEG, PURITY = "diag_log_negativity", "diag_purity"
DESIGN_CONTROLS = ["log_gamma", "log_h", "log_W", "gap_ratio", "channel_ad"]
ALPHA = 0.05 / 3            # Bonferroni over the three confirmatory hypotheses
R_ERGODIC, R_LOCALISED = 0.50, 0.45
SUPPORT_MIN = 0.80          # share of realisations that must pass the echo-state check


def controls(df: pd.DataFrame, with_N: bool = True, purity: bool = False) -> list[str]:
    """Control variables of the primary models: the experimenter-set design variables and
    the gap ratio, plus one indicator per system size beyond the smallest.  Purity is a
    property of the state like the diagnostics themselves, so it enters only in a
    pre-specified sensitivity analysis."""
    out = list(DESIGN_CONTROLS)
    if with_N:
        out += sorted(c for c in df.columns if c.startswith("N_is_"))
    if purity:
        out.append(PURITY)
    return out


# ------------------------------------------------------------------------------ data
def prepare(df: pd.DataFrame, esn: pd.DataFrame, task: str = "narma10", shots: str = "s6",
            diag_prefix: str = "diag_", gap_ref: pd.DataFrame | None = None,
            drop_esp_failures: bool = True) -> pd.DataFrame:
    """Attach the performance gap Delta = NRMSE_ESN(D) - NRMSE_QRC and derived columns.

    diag_prefix="diagmg_" swaps in the diagnostics measured under Mackey-Glass drive.
    gap_ref: reference gap-ratio table (columns N, h, W, r_mean) used to label regimes.
    `common_support` marks sites where at least SUPPORT_MIN of the realisations pass the
    echo-state check under *both* channels; channel comparisons are restricted to them.
    """
    out = df.copy()
    if "site" not in out:                                    # tables written by design v1
        out["site"] = out["cell"].str.replace(r"\|(dephasing|amplitude_damping)", "", regex=True)
    if "esp_ok" in out:
        rate = out.groupby(["site", "channel"])["esp_ok"].mean().unstack()
        ok = (rate >= SUPPORT_MIN).all(axis=1) & rate.notna().all(axis=1) & (rate.shape[1] == 2)
        out["common_support"] = out["site"].map(ok).fillna(False).astype(bool)
        if drop_esp_failures:
            out = out[out["esp_ok"]].copy()
    base = esn[esn["task"] == task].set_index("D")["esn_nrmse"]
    out["nrmse_qrc"] = out[f"nrmse_{task}_{shots}"]
    out["nrmse_qrc_sd"] = out.get(f"nrmse_sd_{task}_{shots}", np.nan)
    out["nrmse_esn"] = out["D"].map(base)
    if out["nrmse_esn"].isna().any():
        raise ValueError(f"no ESN baseline for task {task} at D = {sorted(out.loc[out.nrmse_esn.isna(), 'D'].unique())}")
    out["delta"] = out["nrmse_esn"] - out["nrmse_qrc"]
    out["rel_delta"] = out["delta"] / out["nrmse_esn"]
    if diag_prefix != "diag_":
        for c in [c for c in out.columns if c.startswith(diag_prefix)]:
            out["diag_" + c[len(diag_prefix):]] = out[c]
    out["log_gamma"] = np.log10(out["gamma"])
    out["log_h"] = np.log10(out["h"])
    out["log_W"] = np.log10(out["W"])
    out["channel_ad"] = (out["channel"] == "amplitude_damping").astype(float)
    for n in sorted(out["N"].unique())[1:]:
        out[f"N_is_{n}"] = (out["N"] == n).astype(float)
    if gap_ref is not None:
        ref = gap_ref[gap_ref["N"] == gap_ref["N"].max()].set_index(["h", "W"])["r_mean"]
        out["r_ref"] = [ref.get((h, W), np.nan) for h, W in zip(out["h"], out["W"])]
        out["regime"] = np.select(
            [out["r_ref"] >= R_ERGODIC, (out["r_ref"] <= R_LOCALISED) & (out["W"] >= 10)],
            ["ergodic", "localised"], default="other")
    return out.reset_index(drop=True)


def reliability(df: pd.DataFrame) -> float:
    """Reliability of the per-configuration outcome: the share of the between-configuration
    variance of mean NRMSE that is not sampling error over input realisations,
    1 - mean(sd^2 / n_inputs) / Var(mean NRMSE)."""
    if "nrmse_qrc_sd" not in df or "n_inputs" not in df or df["nrmse_qrc_sd"].isna().all():
        return float("nan")
    err = float((df["nrmse_qrc_sd"] ** 2 / df["n_inputs"]).mean())
    return 1.0 - err / float(df["nrmse_qrc"].var(ddof=1))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph, d = k / n, 1 + z**2 / n
    c, h = (ph + z**2 / (2 * n)) / d, z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------------------ regression core
def _fit_predict(Xtr, ytr, Xte, model="ols"):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    keep = sd > 1e-12                      # drop predictors that are constant in the fitting set
    A = (Xtr[:, keep] - mu[keep]) / sd[keep]
    B = (Xte[:, keep] - mu[keep]) / sd[keep]
    if model == "ols":
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(A)), A], ytr, rcond=None)
        return np.c_[np.ones(len(B)), B] @ beta
    if model == "gbt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                          random_state=0)
        return m.fit(A, ytr).predict(B)
    raise ValueError(model)


def r2(y, yhat) -> float:
    return 1.0 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cv_r2(Xs: list[np.ndarray], y: np.ndarray, groups: np.ndarray, rng, n_splits: int = 5,
          n_repeats: int = 10, model: str = "ols") -> np.ndarray:
    """Grouped, repeated K-fold R^2 for several design matrices on *identical* folds.

    R^2_cv of one repeat is 1 - SSE/SST of the pooled out-of-fold predictions; the result
    is the mean over repeats.  All rows of a group share a fold.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    out = np.zeros((n_repeats, len(Xs)))
    for rep in range(n_repeats):
        fold_of_group = rng.permutation(len(uniq)) % n_splits
        fold = fold_of_group[inv]
        preds = [np.empty_like(y) for _ in Xs]
        for f in range(n_splits):
            te = fold == f
            for m, X in enumerate(Xs):
                preds[m][te] = _fit_predict(X[~te], y[~te], X[te], model)
        out[rep] = [r2(y, p) for p in preds]
    return out.mean(axis=0)


def _cluster_resample(groups: np.ndarray, rng) -> np.ndarray:
    """Row indices of a bootstrap sample of whole groups (drawn with replacement)."""
    uniq = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([rows[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].notna().all()]


# ------------------------------------------------------------------------------ H1a
def nested_gain(df: pd.DataFrame, add: list[str] = (TMI,), base: list[str] = (NEG, COH),
                ctrl: list[str] | None = None, y: str = "delta", margin: float = 0.02,
                alpha: float = ALPHA, n_boot: int = 2000, n_splits: int = 5, n_repeats: int = 10,
                boot_repeats: int | None = None, model: str = "ols", seed: int = 0,
                group: str = "site") -> dict:
    """Gain = R^2_cv(M1) - R^2_cv(M0) with a cluster bootstrap over sites.

    M0: y ~ base + controls;  M1: y ~ base + add + controls.
    H0a: gain <= 0.  H1a: gain > 0, with `margin` the smallest gain of practical interest.
    Decision (pre-specified):
      'supported'    two-sided (1 - alpha) interval lies above 0 AND point estimate >= margin;
      'redundant'    (1 - 2 alpha) interval lies inside (-margin, margin)  (TOST at level alpha);
      'inconclusive' otherwise.
    """
    ctrl = controls(df) if ctrl is None else list(ctrl)
    c0 = available(df, list(base) + ctrl)
    c1 = available(df, list(base) + list(add) + ctrl)
    X0, X1, yv, g = df[c0].to_numpy(float), df[c1].to_numpy(float), df[y].to_numpy(float), df[group].to_numpy()
    rng = np.random.default_rng(seed)
    point = cv_r2([X0, X1], yv, g, rng, n_splits, n_repeats, model)
    boots = np.empty((n_boot, 2))
    br = n_repeats if boot_repeats is None else boot_repeats
    for b in range(n_boot):
        i = _cluster_resample(g, rng)
        boots[b] = cv_r2([X0[i], X1[i]], yv[i], g[i], rng, n_splits, br, model)
    gain = boots[:, 1] - boots[:, 0]
    nan2 = np.array([np.nan, np.nan])
    ci = np.percentile(gain, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if n_boot else nan2
    tost = np.percentile(gain, [100 * alpha, 100 * (1 - alpha)]) if n_boot else nan2
    d = float(point[1] - point[0])
    if not n_boot:
        decision = "no bootstrap"
    elif d >= margin and ci[0] > 0:
        decision = "supported"
    elif tost[0] > -margin and tost[1] < margin:
        decision = "redundant"
    else:
        decision = "inconclusive"
    return dict(n=len(df), n_sites=int(len(np.unique(g))), model=model, added=list(add), alpha=alpha,
                r2_m0=float(point[0]), r2_m1=float(point[1]), gain=d,
                ci_lo=float(ci[0]), ci_hi=float(ci[1]), tost_lo=float(tost[0]), tost_hi=float(tost[1]),
                decision=decision)


def in_sample_r2(df, cols, y="delta") -> float:
    X, yv = df[available(df, cols)].to_numpy(float), df[y].to_numpy(float)
    return r2(yv, _fit_predict(X, yv, X))


def commonality(df: pd.DataFrame, preds=(NEG, COH, TMI), ctrl=None, y="delta") -> pd.DataFrame:
    """Commonality analysis of the variance in y explained by the three diagnostics over and
    above the controls (in-sample R^2 increments; unique and shared components)."""
    ctrl = controls(df) if ctrl is None else list(ctrl)
    r0 = in_sample_r2(df, ctrl, y) if ctrl else 0.0
    inc = {(): 0.0}
    for k in range(1, len(preds) + 1):
        for S in combinations(preds, k):
            inc[S] = in_sample_r2(df, list(S) + ctrl, y) - r0
    a, b, c = preds
    full = inc[(a, b, c)]
    rows = [("controls only (R2)", r0), ("all three (increment)", full),
            (f"unique {a}", full - inc[(b, c)]), (f"unique {b}", full - inc[(a, c)]),
            (f"unique {c}", full - inc[(a, b)]),
            (f"common {a} & {b}", inc[(a, c)] + inc[(b, c)] - inc[(c,)] - full),
            (f"common {a} & {c}", inc[(a, b)] + inc[(b, c)] - inc[(b,)] - full),
            (f"common {b} & {c}", inc[(a, b)] + inc[(a, c)] - inc[(a,)] - full),
            ("common all three", inc[(a,)] + inc[(b,)] + inc[(c,)] - inc[(a, b)] - inc[(a, c)] - inc[(b, c)] + full)]
    return pd.DataFrame(rows, columns=["component", "r2"])


def vif(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in available(df, cols) if df[c].std() > 1e-12]
    out = {}
    for c in cols:
        others = [k for k in cols if k != c]
        out[c] = 1.0 / max(1e-12, 1.0 - in_sample_r2(df, others, y=c))
    return pd.Series(out, name="VIF")


# ------------------------------------------------------------------------------ H1b
def channel_interaction(df: pd.DataFrame, qmi: str = TMI, base=(NEG, COH), ctrl=None,
                        y: str = "delta", bound: float = 0.1, group: str = "site") -> dict:
    """y ~ base + qmi + controls + channel + channel x qmi, everything standardised on the
    pooled data (channel coded 0 = dephasing, 1 = amplitude damping), cluster-robust
    standard errors by site.  The interaction coefficient is the difference between the
    standardised QMI slopes of the two channels.  TOST against +-bound."""
    import statsmodels.api as sm

    ctrl = controls(df) if ctrl is None else list(ctrl)
    cont = [c for c in available(df, list(base) + [qmi] + ctrl)
            if c != "channel_ad" and df[c].std() > 1e-12]
    Z = (df[cont] - df[cont].mean()) / df[cont].std(ddof=0)
    Z["channel_ad"] = df["channel_ad"].to_numpy()
    Z["channel_x_qmi"] = Z["channel_ad"] * Z[qmi]
    yz = (df[y] - df[y].mean()) / df[y].std(ddof=0)
    codes = pd.factorize(df[group])[0]
    fit = sm.OLS(yz, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": codes}, use_t=True)
    b, se = float(fit.params["channel_x_qmi"]), float(fit.bse["channel_x_qmi"])
    dof = len(np.unique(codes)) - 1
    p_lo = 1 - sps.t.cdf((b + bound) / se, dof)     # H0: b <= -bound
    p_hi = sps.t.cdf((b - bound) / se, dof)         # H0: b >= +bound
    cov = fit.cov_params()
    b_ad = float(fit.params[qmi] + b)
    se_ad = float(np.sqrt(cov.loc[qmi, qmi] + cov.loc["channel_x_qmi", "channel_x_qmi"]
                          + 2 * cov.loc[qmi, "channel_x_qmi"]))
    return dict(qmi=qmi, n=len(df), n_sites=dof + 1, interaction=b, se=se,
                p_interaction=float(fit.pvalues["channel_x_qmi"]), tost_bound=bound,
                p_tost=float(max(p_lo, p_hi)),
                slope_dephasing=float(fit.params[qmi]), se_dephasing=float(fit.bse[qmi]),
                slope_amplitude_damping=b_ad, se_amplitude_damping=se_ad,
                r2=float(fit.rsquared))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def h1b(df: pd.DataFrame, variants=(TMI, PAIR), alpha: float = ALPHA, **kw) -> pd.DataFrame:
    """H0b: no channel x QMI interaction.  Holm-corrected over the QMI variants, each test at
    the family level alpha.  'channel-dependent' if the interaction is significant,
    'channel-independent' if TOST places it inside +-bound, 'inconclusive' otherwise."""
    rows = [channel_interaction(df, q, **kw) for q in variants]
    out = pd.DataFrame(rows)
    out["p_interaction_holm"] = holm(out["p_interaction"])
    out["p_tost_holm"] = holm(out["p_tost"])
    out["alpha"] = alpha
    out["decision"] = np.where(out["p_interaction_holm"] < alpha, "channel-dependent",
                               np.where(out["p_tost_holm"] < alpha, "channel-independent", "inconclusive"))
    return out


# ------------------------------------------------------------------------------ H1c
def transfer(src: pd.DataFrame, tgt: pd.DataFrame, cols=None, y: str = "delta", n_boot: int = 2000,
             n_splits: int = 5, n_repeats: int = 10, boot_repeats: int = 2, threshold: float = 0.70,
             alpha: float = ALPHA, seed: int = 0, label: str = "", group: str = "site") -> dict:
    """Fit M1 on the source setting and apply its predictions to the target setting.

    Primary statistic: transfer ratio = r^2_target / R^2_cv,source, where r is the Pearson
    correlation between the source model's predictions and the observed outcome in the
    target (set to 0 if r <= 0).  r^2 is the R^2 the source model reaches in the target after
    a linear recalibration (one intercept and one positive scale), which is needed because
    the outcome has a different mean and spread in each task and system size.
    Also reported: R^2 of the unrecalibrated predictions and the Spearman correlation.
    The model has no system-size terms, so it can be applied to a held-out size.
    """
    if cols is None:
        cols = [NEG, COH, TMI] + controls(src, with_N=False)
    use = [c for c in available(src, list(cols)) if c in tgt.columns]
    Xs, ys, gs = src[use].to_numpy(float), src[y].to_numpy(float), src[group].to_numpy()
    Xt, yt, gt = tgt[use].to_numpy(float), tgt[y].to_numpy(float), tgt[group].to_numpy()
    rng = np.random.default_rng(seed)

    def once(Xs, ys, gs, Xt, yt, reps):
        r_src = cv_r2([Xs], ys, gs, rng, n_splits, reps)[0]
        pred = _fit_predict(Xs, ys, Xt)
        r = np.corrcoef(pred, yt)[0, 1] if pred.std() > 0 else 0.0
        return r_src, max(r, 0.0) ** 2, r2(yt, pred), sps.spearmanr(pred, yt)[0]

    r_src, r_rec, r_raw, rho = once(Xs, ys, gs, Xt, yt, n_repeats)
    ratios, rhos = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i, j = _cluster_resample(gs, rng), _cluster_resample(gt, rng)
        a, c, _, rhos[b] = once(Xs[i], ys[i], gs[i], Xt[j], yt[j], boot_repeats)
        ratios[b] = c / a if a > 0 else np.nan
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lo, hi = (np.nanpercentile(ratios, q) if n_boot else (np.nan, np.nan))
    rlo, rhi = (np.nanpercentile(rhos, q) if n_boot else (np.nan, np.nan))
    return dict(transfer=label, n_source=len(src), n_target=len(tgt), r2_cv_source=float(r_src),
                r2_target_recalibrated=float(r_rec), r2_target_raw=float(r_raw),
                ratio=float(r_rec / r_src) if r_src > 0 else np.nan,
                ratio_lo=float(lo), ratio_hi=float(hi), spearman=float(rho),
                spearman_lo=float(rlo), spearman_hi=float(rhi), alpha=alpha,
                passes=bool(n_boot and lo >= threshold))
