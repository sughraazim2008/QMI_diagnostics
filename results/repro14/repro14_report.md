# Reproduction of Palacios et al. [14] (qualitative)

5 realisations, 10000 steps, IPC to degree 4 and delay 12, readout noise sigma = 0.001.

1. Ergodic regime (W = 0): Spearman correlation between l1 coherence and total IPC (with readout noise) = 0.447 over all runs.

2. bit_flip: relative to the noiseless value, at p_err = 0.002: negativity 0.95, coherence 0.99, IPC 1.00; 0.005: negativity 0.87, coherence 0.96, IPC 1.00; 0.01: negativity 0.76, coherence 0.93, IPC 1.00; 0.02: negativity 0.57, coherence 0.87, IPC 0.99; 0.05: negativity 0.21, coherence 0.71, IPC 0.92.

2. phase_flip: relative to the noiseless value, at p_err = 0.002: negativity 0.95, coherence 0.99, IPC 1.00; 0.005: negativity 0.88, coherence 0.97, IPC 1.00; 0.01: negativity 0.78, coherence 0.93, IPC 1.00; 0.02: negativity 0.60, coherence 0.87, IPC 0.99; 0.05: negativity 0.26, coherence 0.72, IPC 0.94.

3. Collapse: mean residual difference between bit-flip and phase-flip runs around a common quadratic IPC(coherence) curve = -0.061 (IPC range 13.27–15.79).

4. Effect of readout noise on the apparent role of decoherence (mean total IPC):

            ipc_total_noisy  ipc_total_exact
W    p_err                                  
0.0  0.000           15.122           15.370
     0.002           15.113           15.383
     0.005           15.103           15.401
     0.010           15.079           15.443
     0.020           14.973           15.506
     0.050           14.096           15.608
10.0 0.000            7.490            7.990
     0.002            7.823            8.890
     0.005            7.928            9.514
     0.010            7.995           10.139
     0.020            8.035           11.061
     0.050            7.737           12.311
