# CTMC Path Sampling: Modified Rejection, Direct, and Uniformization


This repository contains **working Python implementations** of three strategies to sample continuous-time Markov chain (CTMC) paths conditioned on endpoints over a finite state space:

- **Modified Rejection Sampling** (Nielsen-style conditioning)
- **Direct Sampling** (Hobolth's eigendecomposition approach)
- **Uniformization** (Poisson-subordinated discrete-time chain)

It also includes small driver scripts to compare **computational complexity** and a short example to reproduce figures/timings.

> Context: These implementations accompany a student project/report on endpoint-conditioned CTMC simulation with applications to molecular evolution.

## Repository structure
- Computional_Efficiency_Example.py
- Direct_Sampling_Complexity.py
- Direct_Sampling.py
- Modified_Rejection_Sampling.py
- Rejection_Sampling_Complexity.py
- Uniformization_Complexity.py
- Uniformization.py
- SPA_Project_Report_Paper5.pdf (asset)
- spa_paper.pdf (asset)

## Requirements
- Python 3.9+
- `matplotlib`
- `numpy`


## Quick start

Clone or copy the files into a folder and run any of the scripts directly:

```bash
# Example: run the Nielsen-style modified rejection sampler demo
python Modified_Rejection_Sampling.py

# Example: run the direct sampler demo (eigendecomposition-based)
python Direct_Sampling.py

# Example: run the uniformization sampler demo
python Uniformization.py

# Compare/visualize runtime or step counts (if provided in your version)
python Rejection_Sampling_Complexity.py
python Direct_Sampling_Complexity.py
python Uniformization_Complexity.py

# Small end-to-end example for timing plots (if provided)
python Computional_Efficiency_Example.py
```

> Tip: All scripts assume a **valid generator matrix `Q`** (square, off-diagonal ≥ 0, each row sums to 0). Adjust the hard-coded `Q`, `a`, `b`, `T`, and number of paths in the scripts to match your experiment.


## Algorithms in this repo (what each file does)

- **`Modified_Rejection_Sampling.py`**  
  Implements rejection sampling with *conditioned first jump* when `a != b`. Key pieces typically include:
  - `validate_transition_matrix(Q)`: checks CTMC generator conditions.
  - `sample_waiting_time(rate)`: exponential waiting time.
  - `sample_next_state(current, Q, Qa)`: next state via normalized off-diagonals.
  - `modified_rejection_sampling(Q, a, b, T, max_iterations=...)`: core sampler.
  - `plot_modified_rejection_sampling(...)`: optional visualization.

- **`Direct_Sampling.py`**  
  Implements the *direct* (Hobolth) sampler using eigendecomposition `Q = U D U^{-1}` and closed-forms for first-jump destination and waiting-time:
  - `eigenvalue_decomposition(Q)`
  - `sample_first_state(Q, a, b, T, U, D, U_inv)`
  - `direct_sampling(Q, a, b, T)`
  - `plot_direct_sampling(...)`

- **`Uniformization.py`**  
  Implements uniformization with rate `μ = max_c Q_c` and `R = I + Q/μ`, then samples number of jumps `N ~ Poisson(μT)` conditioned on endpoints and draws intermediate states via powers of `R`:
  - `uniformization_path(Q, a, b, T)` (naming may vary)
  - helper routines to sample `N`, times, and intermediate states
  - optional plotting

- **Complexity scripts** (`*_Complexity.py`)  
  Contain experiments that sweep parameters (e.g., `T`, endpoints) and measure runtime/steps to compare methods. They typically print or plot timing summaries.

- **`Computional_Efficiency_Example.py`**  
  A compact end-to-end example to regenerate comparative timing plots (spelling of file name is intentional—keep as is).


## Tips & gotchas

- **Generator matrix validity**: ensure each row of `Q` sums to zero; negative diagonals; non‑negative off‑diagonals.
- **Small T with a ≠ b**: naïve rejection performs poorly. Use **Modified Rejection**, **Direct**, or **Uniformization**.
- **Large state spaces**: **Direct** and **Uniformization** both need matrix operations; reuse decompositions/powers when sampling many paths.
- **Numerics**: for **Direct**, care with root finding/inverses; for **Uniformization**, cache powers `R^k` if drawing multiple paths.
- **Reproducibility**: set RNG seeds where helpful and log `Q, a, b, T`.


## Theory references (recommended reading)
- Hobolth & Stone (2009), *The Annals of Applied Statistics*, “Simulation from endpoint‑conditioned, continuous‑time Markov chains on a finite state space, with applications to molecular evolution.”  
- Class/project report PDF in this repo summarizing the three samplers and their complexity.

These detail the exact formulae used in **Direct Sampling** (eigendecomposition; conditional first‑jump distribution and waiting‑time density), the **Modified Rejection** conditioning, and **Uniformization** via a Poisson‑subordinated chain.
