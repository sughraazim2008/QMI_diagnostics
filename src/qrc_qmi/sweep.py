"""Run one reservoir configuration end to end, and whole sweeps in parallel with resume."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import SweepConfig, load_n_s, n_s_for
from .diagnostics import DiagnosticAverager
from .hamiltonian import build_hamiltonian, draw_disorder, gap_ratio
from .ipc import ipc
from .readout import add_shot_noise, ridge_select
from .reservoir import Reservoir
from .tasks import mackey_glass_inputs, narma_inputs


def shots_tag(n) -> str:
    return "exact" if n is None else f"s{int(round(np.log10(n)))}"


def make_inputs(cfg: SweepConfig) -> dict:
    """Input sequences shared by every configuration (and by the ESN baseline)."""
    K = cfg.splits.total
    out = {}
    s, targets, disc = narma_inputs(K, cfg.n_inputs, cfg.base_seed, tasks=cfg.narma_tasks)
    out["narma"] = dict(s=s, targets=targets, discarded=disc)
    if cfg.mackey_glass:
        s, targets, _ = mackey_glass_inputs(K, cfg.n_inputs, cfg.base_seed)
        out["mackey_glass"] = dict(s=s, targets=targets, discarded=0)
    # fresh, target-free sequences for the diagnostics window
    rng = np.random.default_rng(np.random.SeedSequence([cfg.base_seed, 404]))
    out["narma"]["s_diag"] = rng.uniform(0.0, 1.0, size=(1, cfg.diag_steps))
    if cfg.mackey_glass:
        sd, _, _ = mackey_glass_inputs(cfg.diag_steps, 1, cfg.base_seed + 1)
        out["mackey_glass"]["s_diag"] = sd
    return out


def _score_family(feats, targets: dict, cfg: SweepConfig, rng, prefix="") -> dict:
    """Ridge readout for every shot level, task and input realisation."""
    row = {}
    names = list(targets)
    for n_shots in cfg.shots:
        tag = shots_tag(n_shots)
        test = np.empty((feats.shape[0], len(names)))
        lam = np.empty_like(test)
        upper = np.empty_like(test)
        for j in range(feats.shape[0]):
            X = add_shot_noise(feats[j], n_shots, rng, model=cfg.noise_model)
            Y = np.stack([targets[t][j] for t in names], axis=1)
            fit = ridge_select(X, Y, cfg.splits)
            test[j], lam[j], upper[j] = fit["test_nrmse"], fit["lambda"], fit["at_upper"]
        for c, t in enumerate(names):
            row[f"{prefix}nrmse_{t}_{tag}"] = float(test[:, c].mean())
            row[f"{prefix}nrmse_sd_{t}_{tag}"] = float(test[:, c].std(ddof=1)) if len(test) > 1 else 0.0
            row[f"{prefix}loglam_{t}_{tag}"] = float(np.log10(lam[:, c]).mean())
            row[f"{prefix}lam_upper_{t}_{tag}"] = float(upper[:, c].mean())
    return row


def run_config(cfg: SweepConfig, point: dict, inputs: dict, n_s_table: dict | None = None) -> dict:
    """Simulate one configuration and return one result row."""
    t0 = time.time()
    N, channel, gamma, h, W = (point[k] for k in ("N", "channel", "gamma", "h", "W"))
    ss = np.random.SeedSequence([cfg.base_seed, *[int(i) for i in point["idx"]]])
    ham_seed, noise_seed, esp_seed, ipc_seed = ss.spawn(4)
    J, D = draw_disorder(N, W, np.random.default_rng(ham_seed), coupling=cfg.coupling)
    H = build_hamiltonian(N, h, J, D, field_prefactor=cfg.field_prefactor)
    n_s = n_s_for(cfg, N, channel, gamma, n_s_table)
    res = Reservoir(H=H, N=N, channel=channel, gamma=gamma, dt=cfg.dt, V=cfg.V, n_s=n_s,
                    encoding=cfg.encoding, propagator=cfg.propagator)
    rng = np.random.default_rng(noise_seed)
    sp = cfg.splits
    row = dict(sweep=cfg.name, N=N, channel=channel, gamma=gamma, h=h, W=W, real=point["real"],
               cell=f"N{N}|{channel}|g{gamma:.6g}|h{h:g}|W{W:g}", D=res.n_features,
               propagator=res.propagator, n_s=n_s if res.propagator == "strang" else 0,
               gap_ratio=gap_ratio(H, N))

    # --- NARMA family (one trajectory serves every NARMA order and the IPC) ---------------
    fam = inputs["narma"]
    feats, rho_end = res.run(fam["s"])
    row.update(_score_family(feats, fam["targets"], cfg, rng))
    if cfg.compute_ipc:
        X = add_shot_noise(feats[0, sp.washout:], cfg.shots[0], rng, model=cfg.noise_model)
        u = 2.0 * fam["s"][0, sp.washout:] - 1.0
        row.update(ipc(X, u, cfg.ipc_degree, cfg.ipc_delay,
                       seed=int(ipc_seed.generate_state(1)[0])))
    # --- diagnostics on a fresh target-free input, starting from a washed-out state --------
    avg = DiagnosticAverager(N, cfg.diag_stride)
    res.run(fam["s_diag"], rho0=rho_end[:1], record=False, on_period_end=avg)
    row.update({f"diag_{k}": v for k, v in avg.mean().items()})
    # --- echo state property ---------------------------------------------------------------
    row["esp_dist"] = res.echo_state_distance(fam["s"][0, :sp.washout], np.random.default_rng(esp_seed))
    row["esp_ok"] = bool(row["esp_dist"] < cfg.esp_tol)

    # --- Mackey-Glass (separate drive, separate diagnostics) ------------------------------
    if cfg.mackey_glass:
        fam = inputs["mackey_glass"]
        feats, rho_end = res.run(fam["s"])
        row.update(_score_family(feats, fam["targets"], cfg, rng))
        avg = DiagnosticAverager(N, cfg.diag_stride)
        res.run(fam["s_diag"], rho0=rho_end[:1], record=False, on_period_end=avg)
        row.update({f"diagmg_{k}": v for k, v in avg.mean().items()})
    row["seconds"] = time.time() - t0
    return row


def provenance(cfg: SweepConfig) -> dict:
    import scipy, sklearn, statsmodels  # noqa: E401
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        sha = ""
    return dict(config=asdict(cfg), config_hash=cfg.hash(), code_version=__version__, git_sha=sha,
                python=platform.python_version(), platform=platform.platform(),
                packages=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                              sklearn=sklearn.__version__, statsmodels=statsmodels.__version__),
                n_s_table=load_n_s())


def run_sweep(cfg: SweepConfig, out_dir: str | Path, n_jobs: int = 4, chunk: int = 24,
              limit: int | None = None) -> pd.DataFrame:
    """Run all configurations, writing a Parquet chunk every `chunk` rows so that an
    interrupted sweep resumes where it stopped.  Final table: <out_dir>/<name>.parquet."""
    from joblib import Parallel, delayed

    out_dir = Path(out_dir)
    chunks = out_dir / "chunks" / f"{cfg.name}-{cfg.hash()}"
    chunks.mkdir(parents=True, exist_ok=True)
    points = list(cfg.grid())[:limit]
    inputs = make_inputs(cfg)
    table = load_n_s()
    (out_dir / f"{cfg.name}.provenance.json").write_text(json.dumps(provenance(cfg), indent=2))
    t0 = time.time()
    for a in range(0, len(points), chunk):
        f = chunks / f"{a:06d}.parquet"
        if f.exists():
            continue
        rows = Parallel(n_jobs=n_jobs)(delayed(run_config)(cfg, p, inputs, table)
                                       for p in points[a:a + chunk])
        pd.DataFrame(rows).to_parquet(f, index=False)
        done = min(a + chunk, len(points))
        print(f"[{cfg.name}] {done}/{len(points)} configurations, {time.time() - t0:.0f} s", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in sorted(chunks.glob("*.parquet"))], ignore_index=True)
    df["narma_discarded"] = inputs["narma"]["discarded"]
    df.to_parquet(out_dir / f"{cfg.name}.parquet", index=False)
    return df
