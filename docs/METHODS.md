# Methods (as implemented)

This is Section 2 of the paper rewritten to match the code in this repository exactly.
Each subsection names the module that implements it. Values marked † are proposals to be
fixed in the pre-registration. Where the code departs from draft v1 of the paper, the
change is listed in [Section 2.12](#212-changes-from-draft-v1-and-open-decisions).

## 2.1 Reservoir model — `hamiltonian.py`, `reservoir.py`

The reservoir is a network of N qubits with all-to-all transverse-field Ising couplings and
on-site disorder [5], [7], [14]:

$$H = \sum_{i<j} J_{ij}\,\sigma^x_i \sigma^x_j + \frac{1}{2}\sum_{i=1}^{N} (h + D_i)\,\sigma^z_i,
\qquad J_{ij} \sim \mathcal{U}\!\left[-\tfrac{J_s}{2}, \tfrac{J_s}{2}\right], \quad D_i \sim \mathcal{U}[-W, W].$$

Energies are in units of J_s = 1 and times in units of 1/J_s (ħ = 1). Every configuration
draws its own J and D, so no Hamiltonian is shared between grid cells.

**Dynamical regime.** H commutes with the parity P = ∏ᵢ σᶻᵢ, because every coupling term flips
two spins. The mean adjacent-gap ratio ⟨r⟩ [Oganesyan and Huse 2007; Atas et al. 2013] is
therefore computed separately in the two parity sectors and averaged; computed on the full
spectrum it would mix two independent level sequences and read as Poisson even for ergodic
dynamics. Reference values are 0.531 (GOE) and 0.386 (Poisson). Because the sectors of N = 4
and N = 6 hold only 8 and 32 levels, ⟨r⟩ of a single small Hamiltonian is too noisy to
classify a regime. Regime labels are therefore assigned per (h, W) point from a reference
ensemble at N = 10 (100 realisations, `scripts/pilot_gap_ratio.py`): *ergodic* if
⟨r⟩ ≥ 0.50, *localised* if ⟨r⟩ ≤ 0.45 and W ≥ 10, *other* otherwise. The per-configuration
gap ratio is still recorded and entered as a control.

**Input injection.** At step k a scalar input s_k ∈ [0, 1] is written into qubit 1 by the
reset-and-write map [5]:

$$\rho \;\mapsto\; |\psi_{s_k}\rangle\langle\psi_{s_k}| \otimes \mathrm{Tr}_1(\rho),
\qquad |\psi_{s}\rangle = \sqrt{1-s}\,|0\rangle + \sqrt{s}\,|1\rangle ,$$

followed by evolution for one input period Δt under the master equation of Section 2.2.
Two further encodings are implemented: the mixed state ρ₁ = diag(s, 1 − s), which is the
encoding of the main text of [14] and is used for the reproduction, and the rotation
|ψ_s⟩ = exp(−iπsσˣ/2)|0⟩, used as the sensitivity check.

## 2.2 Decoherence channels and simulation — `channels.py`, `reservoir.py`

Decoherence acts locally on every qubit through a Lindblad master equation:

$$\dot{\rho} = -i[H, \rho] + \sum_{i=1}^{N} \left( L_i \rho L_i^{\dagger} - \tfrac{1}{2}\{L_i^{\dagger} L_i, \rho\} \right).$$

| Channel | Jump operator | Unital | Kraus parameter after time t | Role |
|---|---|---|---|---|
| Pure dephasing | L = (√γ/2) σᶻ | Yes | p = (1 − e^{−γt/2})/2 | Main comparison (H1b) |
| Amplitude damping | L = √γ σ⁻, σ⁻ = \|0⟩⟨1\| | No | p = 1 − e^{−γt} | Main comparison (H1b) |
| Phase flip, bit flip | L = √γ σᶻ, √γ σˣ | Yes | p = (1 − e^{−2γt})/2 | Reproduction of [14] only |

With these definitions pure dephasing and amplitude damping both damp single-qubit coherences
as e^{−γt/2} (T₂ = 2/γ), so the main comparison is at equal coherence decay. Amplitude damping
additionally relaxes populations at rate γ (T₁ = 1/γ) towards |0⟩.

States are dense 2ᴺ × 2ᴺ density matrices. Each input period is divided into V sub-intervals
of length τ = Δt/V, at the end of which the readout is taken (Section 2.3). Within a
sub-interval:

* **N ≤ 4:** the exact propagator exp(𝓛τ) of the vectorised Liouvillian is applied.
* **N ≥ 5:** n_s second-order (Strang) steps of length δ = τ/n_s are applied:
  half a step of the local channel, the unitary exp(−iHδ), then half a step of the channel.
  Each local channel is the exact solution of its single-qubit dissipator, applied through its
  Kraus form.

n_s is calibrated per channel and γ (`scripts/calibrate_ns.py`, results in `configs/n_s.json`)
as the smallest power of two for which the Strang state stays within trace distance 10⁻⁴ of
the exact propagator after 100 input periods, in the worst case over the (h, W) grid and two
Hamiltonian realisations at N = 4 and N = 5. For N > 5 the calibrated n_s is doubled as a
safety margin. The measured values are in [Section 2.10](#210-validation).

## 2.3 Readout and finite sampling — `reservoir.py`, `readout.py`

At the end of each of the V sub-intervals the expectation values ⟨σᶻᵢ⟩ and ⟨σᶻᵢσᶻⱼ⟩ are
recorded (time multiplexing) [5], [8]. This gives D = V [N + N(N − 1)/2] features plus a bias:
D = 100, 210 and 360 for N = 4, 6 and 8 with V = 10†, Δt = 10†.

Finite sampling is part of every primary analysis. Each expectation value ⟨O⟩ ∈ [−1, 1] is
replaced by ⟨O⟩ + ε with ε ~ 𝒩(0, (1 − ⟨O⟩²)/n_shots), the Gaussian approximation to the mean
of n_shots outcomes ±1, clipped to [−1, 1]. Noise is drawn independently for every observable
and time step, in training, validation and test data alike. The primary setting is
n_shots = 10⁶, close to the σ = 10⁻³ of [14]; 10⁵ and 10⁴ are secondary. Two simplifications
are made and are discussed as limitations: measurement back-action is not modelled (an
ensemble or restart protocol is assumed [28]), and all observables are treated as separately
estimated, although in practice they would be computed from the same z-basis shots and their
errors would be correlated.

## 2.4 Tasks and capacity — `tasks.py`, `ipc.py`

**NARMA-10** [3] is the primary task. With u_k drawn independently from 𝒰[0, 0.5]:

$$y_{k+1} = 0.3\,y_k + 0.05\,y_k \sum_{i=0}^{9} y_{k-i} + 1.5\,u_{k-9}\,u_k + 0.1 .$$

The reservoir receives s_k = 2u_k and the readout at step k is trained on y_{k+1}, which
depends only on inputs up to u_k. A realisation is redrawn if any NARMA series diverges on
it; the number discarded is stored in the result table. Each realisation has a washout of 500
steps, then 2,000 training, 500 validation and 1,000 test steps†. Five input realisations are
shared by all configurations and by the classical baseline.

**Transfer tasks.** NARMA-5 uses the same equation with the sum over five terms and the
product u_{k−4}u_k. NARMA-20 uses the bounded variant of Rodan and Tiňo (2011),
y_{k+1} = tanh(0.3 y_k + 0.05 y_k Σ_{i=0}^{19} y_{k−i} + 1.5 u_{k−19} u_k + 0.01). All three
NARMA orders are scored on the same reservoir trajectory, since they share the input.
**Mackey–Glass** [ref to add: Mackey and Glass 1977] uses ẋ = 0.2 x(t − 17)/(1 + x(t − 17)¹⁰) − 0.1 x,
integrated with a fourth-order Runge–Kutta scheme (step 0.1) and sampled every time unit after
a transient of 1,000; the input is s_k = (x_k − 0.2)/1.2 and the target is x_{k+1}.

**Information processing capacity** follows [4]. Targets are products of normalised Legendre
polynomials of delayed inputs u = 2s − 1, up to total degree 3 and delay 20† (2,023 targets).
The capacity of a target z is ‖Qᵀz‖²/‖z‖², with Q an orthonormal basis of the centred
features, evaluated on the 3,500 post-washout steps of the first input realisation with the
primary shot noise. Capacities not exceeding the largest capacity obtained from a
time-shuffled copy of the input are set to zero. With series this short the threshold is
about 0.04–0.05 per target, so weak capacities are lost and the total is a lower bound; it is
used as a secondary covariate, not as an absolute capacity.

## 2.5 Readout and classical baseline — `readout.py`, `esn.py`

All readouts are ridge regressions with an unpenalised bias, minimising
‖y − Xw − b‖² + λ‖w‖² on the training segment. Features are centred with training means and
not rescaled. λ is chosen from 21 log-spaced values between 10⁻¹⁰ and 1 by validation NRMSE,
and the training-segment fit at that λ is scored on the test segment:

$$\mathrm{NRMSE} = \sqrt{ \langle (y_k - \hat{y}_k)^2 \rangle / \mathrm{Var}(y) } .$$

The classical baseline is a leaky echo state network [1], [43]:

$$x_{k+1} = (1-a)\,x_k + a \tanh\!\left( W x_k + W_{\mathrm{in}}\, s_k + b \right).$$

It has D nodes, so both models fit D + 1 readout weights [32], [33]. W has 10% non-zero
entries drawn from 𝒰[−1, 1] and is rescaled to spectral radius ρ; W_in and b are drawn from
𝒰[−1, 1] and multiplied by the input and bias scalings. The ESN is tuned with a
tree-structured Parzen estimator (Optuna) over a ∈ [0.05, 1] (log scale), ρ ∈ [0.1, 1.5],
input scaling ∈ [0.01, 2] (log scale) and bias scaling ∈ [0, 1], with 100 trials per
(task, D)†. The objective is validation NRMSE averaged over the five input realisations and
three tuning seeds. The selected setting is then scored on the test segment with ten fresh
weight seeds. ESN features are noiseless.

The quantum reservoir receives no comparable tuning: each configuration is fixed and only λ
is fitted. Two reference baselines are reported: an untuned ESN (a = 1, ρ = 0.9, unit input
scaling, no bias) and a ridge regression on the last ten inputs and their pairwise products.

## 2.6 Performance gap — `stats.prepare`

For configuration c with readout dimension D_c,

$$\Delta_c = \overline{\mathrm{NRMSE}}_{\mathrm{ESN}}(D_c) - \overline{\mathrm{NRMSE}}_{\mathrm{QRC}}(c),$$

with means over the five input realisations (and the ten weight seeds for the ESN).
Δ_c > 0 means the quantum reservoir is better. Δ_c / NRMSE_ESN is a secondary outcome.

## 2.7 State diagnostics — `diagnostics.py`

Diagnostics are computed from the reservoir state alone. Starting from the washed-out state
at the end of the first input realisation, the reservoir is driven by a fresh sequence of 200
inputs from the task's input distribution (uniform for the NARMA family, a separate
Mackey–Glass segment for that task). Each diagnostic is evaluated on the state at the end of
every period, just before the next reset, and averaged over the window. Entropies are in bits.

| Diagnostic | Definition | Notes |
|---|---|---|
| Negativity N̄ | Mean of (‖ρ^{T_A}‖₁ − 1)/2 over all 2^{N−1} − 1 bipartitions [36] | As in [14]; mean log-negativity log₂‖ρ^{T_A}‖₁ is secondary |
| l1 coherence C | Σ_{i≠j} \|ρ_ij\| / (2ᴺ − 1), computational basis [37] | Basis matches the readout |
| Total mutual information T | [Σᵢ S(ρᵢ) − S(ρ)] / N [14], [38] | Primary QMI |
| Pairwise QMI Ī | Mean over pairs of S(ρᵢ) + S(ρⱼ) − S(ρᵢⱼ) | Secondary QMI; half-chain I(A:B) also stored |
| Classical correlations, hookup | T(Δ[ρ]) and Σᵢ H(diag ρᵢ) − S(ρ) [39] | Stored for the reproduction of [14] |
| Controls | Purity Tr ρ², S(ρ)/N, log₁₀ γ, log₁₀ h, log₁₀ W, gap ratio r, N, channel | Entered in every model; a control that is constant in a fitting set is dropped |

For pure states the mutual information across a bipartition equals twice the entanglement
entropy [40], so any difference between QMI and entanglement measures comes from mixedness,
which the purity and entropy controls track.

## 2.8 Experimental design — `config.py`

The unit of analysis is a configuration: one Hamiltonian realisation at one grid point. A
grid cell is the set of realisations sharing (N, channel, γ, h, W); cells are the groups for
cross-validation and the clusters for the bootstrap.

| Factor | Main sweep (`main`) | Transfer sweep (`transfer`) | Reproduction of [14] |
|---|---|---|---|
| N | 4, 6 | 8 | 5 |
| Channel | Dephasing, amplitude damping | Same | Bit flip, phase flip |
| γ (units of J_s) | 8 log-spaced values, 10⁻³ to 1† | Every second main value | p_err from 0 to 0.05 |
| h | 0.1, 1, 10† | Same | 1 |
| W | 0.1, 5, 20† | Same | 0, 10 |
| Realisations per cell | 10 | 5 | 5 (default) |
| Configurations | 2,880 | 360 | 120 |
| Outputs | NARMA-10, IPC, diagnostics | NARMA-5/10/20, Mackey–Glass, diagnostics | IPC, diagnostics |

The h and W levels were chosen from the gap-ratio pilot ([Section 2.10](#210-validation)).

## 2.9 Statistical analysis — `stats.py`, `scripts/analyze.py`

Predictors are standardised within each fitting set. Cross-validation is 5-fold, grouped by
grid cell, and repeated 10 times; R²_cv of one repeat is 1 − SSE/SST of the pooled
out-of-fold predictions, and repeats are averaged. Uncertainty comes from a cluster bootstrap
over cells (2,000 resamples), recomputing the full cross-validation in each resample, with
percentile intervals. Each primary hypothesis is tested at α = 0.05.

**H1a (added value of QMI).** Two linear models of Δ are compared on identical folds:

$$M_0: \ \Delta \sim \bar{N} + C + \text{controls}, \qquad M_1: \ \Delta \sim \bar{N} + C + T + \text{controls}.$$

The statistic is ΔR²_cv = R²_cv(M₁) − R²_cv(M₀). H1a is supported if ΔR²_cv ≥ 0.02† and its
95% lower bound exceeds 0. QMI is declared redundant if the 90% interval lies within ±0.02
(TOST at α = 0.05). Otherwise the result is inconclusive. The test is run pooled, within each
channel, within the ergodic and localised regimes, and within each N. Sensitivity analyses
swap T for pairwise QMI, use log-negativity, use the relative gap, repeat the comparison at
10⁵ and 10⁴ shots, and repeat it with gradient-boosted trees (fewer bootstrap resamples).
The same nested comparison is also run with coherence and with negativity as the added
predictor, to show what each diagnostic contributes. Spearman correlations, variance
inflation factors and a commonality analysis show how much explained variance is shared.

**H1b (channel dependence).** On the pooled data, with Δ and all continuous predictors
standardised:

$$\Delta \sim \bar{N} + C + T + \text{controls} + \text{channel} + \text{channel} \times T .$$

The channel × T coefficient (amplitude-damping slope minus dephasing slope) is tested with
standard errors clustered by cell and a t reference distribution with (number of cells − 1)
degrees of freedom, Holm-corrected across the two QMI variants. Channel-independence is
claimed only if TOST places the interaction within ±0.1†. Interactions of channel with C and
N̄ are reported as secondary.

**H1c (transfer).** M₁ is fitted on a source setting and applied unchanged, with the source's
standardisation, to a target setting. Three transfers are tested: between channels (both
directions), from N = 4 and 6 to N = 8, and from NARMA-10 to each second task on the
configurations of the transfer sweep. The transfer ratio is R²_target / R²_cv,source, where
R²_target is measured against the target's own mean. H0c is rejected when the 95% lower
bootstrap bound of the ratio is at least 0.70 in every transfer. The Spearman correlation
between predicted and observed Δ is reported as a secondary measure, because Δ shifts in
scale between tasks and sizes.

## 2.10 Validation — `tests/`, `scripts/`

The main sweep starts only after these checks pass.

1. **Unit tests** (`pytest`, 54 tests). A Bell pair gives negativity 0.5 and T = 2 bits;
   |+⟩^{⊗N} gives C = 2ᴺ − 1 and T = 0; product and classically correlated states give zero
   negativity; a GHZ state gives negativity 0.5 on every cut; each channel's Kraus form equals
   the exponential of its Lindblad generator, its Choi matrix is positive semidefinite and
   trace-preserving; amplitude damping relaxes to |0⟩; dephasing and amplitude damping give the
   same coherence decay; the fast channel code equals an explicit Kraus sum.
2. **Integrator accuracy.** The Strang error falls as n_s⁻²; the exact and Strang propagators
   agree with an independent adaptive Runge–Kutta solution of the master equation; n_s is set
   by the calibration of Section 2.2 (measured values below).
3. **Echo state property.** For every configuration, two Haar-random initial states driven by
   the same washout input must end within trace distance 10⁻⁶. Failures are counted, reported
   by channel and γ, and excluded.
4. **Reproduction of [14]** (`scripts/reproduce_palacios.py`): at N = 5 under bit-flip and
   phase-flip noise, the IPC–coherence relation is monotonic, negativity falls substantially
   before capacity drops, and the two channels collapse onto one curve.
5. **Baseline sanity.** The tuned ESN must beat the untuned ESN and the linear–quadratic
   reference.

<!-- MEASURED-VALUES -->

## 2.11 Reproducibility — `sweep.py`

Hypotheses, margins, the parameter grid and the analysis plan are pre-registered after the
pilot and before the main sweep. Every random draw derives from one base seed through
`numpy.random.SeedSequence` keyed by the grid indices, so any single configuration can be
recomputed bit for bit (tested). Each sweep writes a provenance file with its configuration,
configuration hash, git commit, package versions and the n_s table. Results are stored as
Parquet tables, and every figure and table is produced from them by `scripts/analyze.py`.
Exact package versions are pinned in `requirements-lock.txt`.

## 2.12 Changes from draft v1 and open decisions

Changes already made in the code, to be carried into the paper text:

1. **Gap ratio within parity sectors** (2.1). Draft v1 computes ⟨r⟩ "of H"; that would give
   Poisson statistics in every regime because of the Z₂ symmetry.
2. **Regime labels from an N = 10 reference ensemble** (2.1), because ⟨r⟩ cannot resolve a
   regime at N = 4 or 6.
3. **"Chain" → "network"** (2.1): the couplings are all-to-all.
4. **The encoding of [14] is the mixed state diag(s, 1 − s)**, not the pure state of [5]; the
   draft cites both for the pure-state map. The reproduction uses the mixed encoding.
5. **[14] uses continuous Lindblad noise** γ(σρσ − ρ) quoted as an error probability, not
   discrete Pauli channels; the code follows the Lindblad form.
6. **Exact propagator for N ≤ 4, Strang for N ≥ 5** (2.2), with n_s calibrated at N = 4 and 5.
7. **log₁₀ h and log₁₀ W as controls** (2.7), because the levels are log-spaced; S(ρ)/N in
   place of S(ρ) so the control is comparable across N; channel added as a control in pooled
   models.
8. **Task transfer is evaluated within the transfer sweep** (2.9), so that it is not
   confounded with the size transfer. Draft v1 does not say which configurations are used.

Decisions that remain yours and should be settled in the pre-registration:

1. **Δ and NRMSE are the same outcome within one N.** The ESN term in Δ depends only on
   (task, D), so within a fixed N it is a constant, and because N is a control, every
   regression of Δ has exactly the same R² as a regression of −NRMSE_QRC. The gap framing
   matters for the *sign* of Δ and for comparisons across N and tasks, but H1a and H1b as
   written test whether QMI predicts quantum-reservoir error. Either say so, or make the
   baseline vary by configuration (for example an ESN matched to each configuration's measured
   rank or IPC).
2. **H1a decision rule.** The hypothesis table says "H0a: the gain is below 0.02"; Section 2.9
   supports H1a when the point estimate is ≥ 0.02 and the lower bound is > 0. These are
   different tests. The code implements the second and also reports whether the lower bound
   itself reaches 0.02.
3. **Echo-state exclusions** remove whole regions of the grid (weak decoherence), so the
   analysed sample is not the designed sample. Options: longer washout, a looser tolerance
   justified against the shot-noise floor, or keeping the exclusion and stating its extent.
4. **Mackey–Glass one-step prediction at unit sampling** is close to trivial for any model
   with a linear readout; consider a longer horizon or coarser sampling.
5. **Ridge range.** Check the share of fits at λ = 1 in the report; extend the range if it is
   not negligible.
6. **Compute budget** for N = 6 and N = 8 (see the measured values above).
