"""Validation check 4 (Methods 2.10): qualitative reproduction of Palacios et al. [14].

Set-up as reported for [14] (arXiv:2409.17734) -- CONFIRM EACH VALUE AGAINST THE PAPER:
  N = 5, all-to-all H = sum J_ij X_i X_j + sum (h + w_i) Z_i, J_ij ~ U[0, 1], w_i ~ U[-W, W],
  h = 1, W = 0 (ergodic) or 10 (localised), dt = 10, mixed input rho_1 = diag(s, 1 - s),
  bit-flip / phase-flip Lindblad noise gamma (sigma rho sigma - rho) on every qubit, strength
  quoted as p_err = (1 - exp(-2 gamma dt)) / 2 in [0, 0.05], Gaussian readout noise sigma = 1e-3.
Differences from [14], chosen to keep the run short: Z + ZZ readout without time
multiplexing, IPC up to degree 4 and delay 12 on 10^4 steps (the paper uses degree 6 and
10^5 steps), fewer disorder realisations.  The comparison is therefore qualitative.

Usage: python scripts/reproduce_palacios.py [--n-real 5] [--steps 10000] [--jobs 6]
"""
import argparse
import os

for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from qrc_qmi import figures as fg  # noqa: E402
from qrc_qmi.diagnostics import DiagnosticAverager  # noqa: E402
from qrc_qmi.hamiltonian import build_hamiltonian, draw_disorder, gap_ratio  # noqa: E402
from qrc_qmi.ipc import ipc  # noqa: E402
from qrc_qmi.readout import add_shot_noise  # noqa: E402
from qrc_qmi.reservoir import Reservoir  # noqa: E402

N, DT, WASHOUT = 5, 10.0, 1000
P_ERR = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)


def one(channel, p_err, W, r, steps, n_s, degree, delay, sigma):
    ss = np.random.SeedSequence([14, int(W), r])            # same Hamiltonian across noise settings
    J, D = draw_disorder(N, W, np.random.default_rng(ss), coupling="positive")
    H = build_hamiltonian(N, 1.0, J, D, field_prefactor=1.0)
    gamma = -np.log(1 - 2 * p_err) / (2 * DT) if p_err > 0 else 0.0
    res = Reservoir(H=H, N=N, channel=channel if p_err > 0 else "none", gamma=gamma, dt=DT, V=1,
                    n_s=n_s, encoding="mixed_z", propagator="strang")
    rng = np.random.default_rng(np.random.SeedSequence([15, r]))
    u = rng.uniform(-1, 1, size=WASHOUT + steps)
    feats, rho = res.run((u[None, :] + 1) / 2)
    out = dict(channel=channel, p_err=p_err, W=W, real=r, gap_ratio=gap_ratio(H, N))
    for label, sd in (("noisy", sigma), ("exact", 0.0)):
        X = feats[0, WASHOUT:]
        X = add_shot_noise(X, None, rng, model="gaussian", sigma=sd) if sd > 0 else X
        out.update({f"{k}_{label}": v for k, v in ipc(X, u[WASHOUT:], degree, delay, seed=r).items()})
    avg = DiagnosticAverager(N)
    res.run(rng.uniform(0, 1, size=(1, 100)), rho0=rho, record=False, on_period_end=avg)
    out.update(avg.mean())
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=5)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--n-s", type=int, default=128, help="Strang steps per input period")
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--delay", type=int, default=12)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out", default="results/repro14")
    a = p.parse_args()
    jobs = [(c, pe, W, r, a.steps, a.n_s, a.degree, a.delay, a.sigma)
            for c in ("bit_flip", "phase_flip") for pe in P_ERR for W in (0.0, 10.0) for r in range(a.n_real)]
    df = pd.DataFrame(Parallel(n_jobs=a.jobs)(delayed(one)(*j) for j in jobs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "repro14.parquet", index=False)

    m = df.groupby(["W", "channel", "p_err"])[["ipc_total_noisy", "ipc_total_exact", "coherence",
                                               "negativity", "tmi", "classical_corr", "gap_ratio"]].mean()
    m.to_csv(out / "repro14_means.csv")
    print(m.round(4).to_string())
    erg = df[df.W == 0]
    lines = ["# Reproduction of Palacios et al. [14] (qualitative)\n",
             f"{a.n_real} realisations, {a.steps} steps, IPC to degree {a.degree} and delay {a.delay}, "
             f"readout noise sigma = {a.sigma}.\n"]
    rho = spearmanr(erg["coherence"], erg["ipc_total_noisy"])[0]
    lines.append(f"1. Ergodic regime (W = 0): Spearman correlation between l1 coherence and total IPC "
                 f"(with readout noise) = {rho:.3f} over all runs.\n")
    me = erg.groupby(["channel", "p_err"])[["ipc_total_noisy", "coherence", "negativity"]].mean()
    for ch in ("bit_flip", "phase_flip"):
        t = me.loc[ch]
        rel = t / t.loc[0.0]
        lines.append(f"2. {ch}: relative to the noiseless value, at p_err = "
                     + "; ".join(f"{pe:g}: negativity {rel.loc[pe, 'negativity']:.2f}, coherence "
                                 f"{rel.loc[pe, 'coherence']:.2f}, IPC {rel.loc[pe, 'ipc_total_noisy']:.2f}"
                                 for pe in P_ERR[1:]) + ".\n")
    # collapse: after a monotone fit of IPC on coherence, does channel explain residual variance?
    z = np.polyfit(erg["coherence"], erg["ipc_total_noisy"], 2)
    resid = erg["ipc_total_noisy"] - np.polyval(z, erg["coherence"])
    gap = resid[erg.channel == "bit_flip"].mean() - resid[erg.channel == "phase_flip"].mean()
    lines.append(f"3. Collapse: mean residual difference between bit-flip and phase-flip runs around a common "
                 f"quadratic IPC(coherence) curve = {gap:+.3f} (IPC range {erg['ipc_total_noisy'].min():.2f}"
                 f"–{erg['ipc_total_noisy'].max():.2f}).\n")
    mm = df.groupby(["W", "p_err"])[["ipc_total_noisy", "ipc_total_exact"]].mean()
    lines.append("4. Effect of readout noise on the apparent role of decoherence (mean total IPC):\n\n"
                 + mm.round(3).to_string() + "\n")
    (out / "repro14_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    d = erg.rename(columns={"coherence": "diag_coherence", "negativity": "diag_negativity",
                            "tmi": "diag_tmi_per_qubit", "ipc_total_noisy": "ipc_total"})
    d["diag_tmi_per_qubit"] = d["diag_tmi_per_qubit"] / N
    fg.scatter_by_channel(d, ["diag_coherence", "diag_negativity", "diag_tmi_per_qubit"], "ipc_total",
                          out / "repro14_ipc_vs_diagnostics.png",
                          "Reproduction of [14], ergodic regime: IPC against diagnostics")
