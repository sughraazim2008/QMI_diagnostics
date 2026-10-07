"""Figures.  Every figure is produced from the result tables by a function here."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as st  # noqa: E402

COLORS = {"dephasing": "#2a78d6", "amplitude_damping": "#eb6834",
          "phase_flip": "#2a78d6", "bit_flip": "#eb6834"}
LABELS = {"dephasing": "Pure dephasing", "amplitude_damping": "Amplitude damping",
          "phase_flip": "Phase flip", "bit_flip": "Bit flip"}
MARKERS = {"dephasing": "o", "amplitude_damping": "s", "phase_flip": "o", "bit_flip": "s"}
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
NICE = {st.NEG: "Mean negativity", st.COH: "l1 coherence (normalised)",
        st.TMI: "Total mutual information / N (bits)", st.PAIR: "Pairwise QMI (bits)",
        "diag_purity": "Purity", "delta": "Gap Δ = NRMSE(ESN) − NRMSE(QRC)",
        "nrmse_qrc": "QRC NRMSE", "ipc_total": "Total IPC"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": "#52514e",
    "ytick.labelcolor": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 200,
})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _channels(df):
    return [c for c in COLORS if c in set(df["channel"])][:2]


def scatter_by_channel(df, xs, y, path, title, zero_line=False):
    fig, axes = plt.subplots(1, len(xs), figsize=(3.6 * len(xs), 3.3), sharey=True)
    for ax, x in zip(np.atleast_1d(axes), xs):
        for ch in _channels(df):
            d = df[df["channel"] == ch]
            ax.scatter(d[x], d[y], s=9, alpha=0.45, color=COLORS[ch], marker=MARKERS[ch],
                       linewidths=0, label=LABELS[ch])
        if zero_line:
            ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel(NICE.get(x, x))
    np.atleast_1d(axes)[0].set_ylabel(NICE.get(y, y))
    np.atleast_1d(axes)[0].legend(loc="best", markerscale=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


def lines_vs_gamma(df, ys, path, title, by="regime"):
    """Median and interquartile band against the decoherence rate, one panel per quantity."""
    groups = [g for g in ("ergodic", "other", "localised") if g in set(df.get(by, []))] or [None]
    fig, axes = plt.subplots(len(groups), len(ys), figsize=(3.4 * len(ys), 2.6 * len(groups)),
                             sharex=True, squeeze=False)
    for r, g in enumerate(groups):
        sub = df if g is None else df[df[by] == g]
        for c, y in enumerate(ys):
            ax = axes[r, c]
            for ch in _channels(df):
                q = sub[sub["channel"] == ch].groupby("gamma")[y].quantile([0.25, 0.5, 0.75]).unstack()
                if q.empty:
                    continue
                ax.fill_between(q.index, q[0.25], q[0.75], color=COLORS[ch], alpha=0.15, linewidth=0)
                ax.plot(q.index, q[0.5], color=COLORS[ch], marker=MARKERS[ch], ms=4, label=LABELS[ch])
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(NICE.get(y, y), fontsize=9)
            if c == 0:
                ax.set_ylabel(g or "")
            if r == len(groups) - 1:
                ax.set_xlabel("Decoherence rate γ (units of J_s)")
    axes[0, 0].legend(loc="best")
    fig.suptitle(title + "  (median, interquartile band)", x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


def correlation_heatmap(df, cols, path, title):
    C = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.2, 0.62 * len(cols) + 1.4))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [c.replace("diag_", "") for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), names)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = C.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
    ax.set_title(title)
    return _save(fig, path)


def forest(table: pd.DataFrame, path, title, margin=0.02, value="gain", lo="ci95_lo", hi="ci95_hi",
           xlabel="Gain in cross-validated R² from adding QMI (95% bootstrap interval)"):
    t = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(t) + 1.3))
    ax.axvspan(-margin, margin, color="#f0efec", zorder=0, label=f"equivalence region ±{margin}")
    ax.axvline(0, color=INK, lw=0.8)
    ax.errorbar(t[value], range(len(t)), xerr=[t[value] - t[lo], t[hi] - t[value]], fmt="o",
                color="#2a78d6", ecolor="#2a78d6", elinewidth=1.5, ms=5, capsize=0)
    ax.set_yticks(range(len(t)), t["label"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title)
    return _save(fig, path)


def predicted_vs_observed(pairs: dict, path, title):
    """pairs: {label: (predicted, observed)}"""
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.3 * len(pairs), 3.2), squeeze=False)
    for ax, (label, (p, o)) in zip(axes[0], pairs.items()):
        ax.scatter(p, o, s=8, alpha=0.5, color="#2a78d6", linewidths=0)
        lim = [min(p.min(), o.min()), max(p.max(), o.max())]
        ax.plot(lim, lim, color=INK, lw=0.8)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Predicted Δ (source model)")
    axes[0, 0].set_ylabel("Observed Δ (target)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


def gap_ratio_map(gap: pd.DataFrame, path):
    Ns = sorted(gap["N"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.0 * len(Ns) + 0.8, 3.0), squeeze=False)
    for ax, N in zip(axes[0], Ns):
        P = gap[gap["N"] == N].pivot(index="h", columns="W", values="r_mean")
        im = ax.imshow(P, origin="lower", cmap="Blues", vmin=0.386, vmax=0.531, aspect="auto")
        ax.set_xticks(range(P.shape[1]), [f"{w:g}" for w in P.columns], rotation=45)
        ax.set_yticks(range(P.shape[0]), [f"{h:g}" for h in P.index])
        ax.set_title(f"N = {N}", fontsize=9)
        ax.set_xlabel("Disorder W")
        ax.grid(False)
    axes[0, 0].set_ylabel("Field h")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, label="⟨r⟩  (Poisson 0.386 → GOE 0.531)")
    fig.suptitle("Mean adjacent-gap ratio within parity sectors", x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)
