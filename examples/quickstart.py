"""Quickstart: simulate one noisy quantum reservoir, score it on NARMA-10, and read off
its state diagnostics -- in under 25 lines. Run with:

    python -m pip install -e .
    python examples/quickstart.py
"""
import numpy as np

from qrc_qmi.diagnostics import all_diagnostics
from qrc_qmi.hamiltonian import build_hamiltonian, draw_disorder
from qrc_qmi.readout import Splits, ridge_select
from qrc_qmi.reservoir import Reservoir
from qrc_qmi.tasks import narma_inputs

N = 4                                   # number of qubits
rng = np.random.default_rng(0)
J, D = draw_disorder(N, W=5.0, rng=rng)          # random couplings + on-site disorder
H = build_hamiltonian(N, h=1.0, J=J, D=D)        # transverse-field Ising Hamiltonian

reservoir = Reservoir(H=H, N=N, channel="dephasing", gamma=0.05)  # pure dephasing, rate 0.05

splits = Splits(washout=200, n_train=800, n_val=200, n_test=400)  # short demo run
s, targets, _ = narma_inputs(splits.total, n_real=1, seed=1, tasks=("narma10",))
features, rho_final = reservoir.run(s)           # drive the reservoir with the NARMA input

fit = ridge_select(features[0], targets["narma10"][0], splits)  # ridge-regression readout
print(f"NARMA-10 test NRMSE: {fit['test_nrmse'][0]:.4f}")

diagnostics = all_diagnostics(rho_final[0], N)   # state diagnostics, no task targets used
print(f"Negativity: {diagnostics['negativity']:.4f}   "
      f"l1 coherence: {diagnostics['coherence']:.4f}   "
      f"QMI/qubit: {diagnostics['tmi_per_qubit']:.4f}")
