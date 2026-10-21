# PH Hankel / Schatten-p research reproducibility package

Research demo: stable passive port-Hamiltonian model order reduction using Hankel-operator Schatten-p objectives. This is **not** a published method or a claimed globally optimal solver.

Requirements: Python >=3.10, NumPy, SciPy, Matplotlib.

Run (recommended single-thread BLAS):

```
OPENBLAS_NUM_THREADS=1 python experiment.py
OPENBLAS_NUM_THREADS=1 python stress.py
OPENBLAS_NUM_THREADS=1 python rankone.py
OPENBLAS_NUM_THREADS=1 python intrinsic.py
OPENBLAS_NUM_THREADS=1 python margin.py
OPENBLAS_NUM_THREADS=1 python tradeoffs.py
OPENBLAS_NUM_THREADS=1 python audit.py
```

- `experiment.py`: pH systems, controllability/observability Gramians, analytically differentiated even Schatten-p and H2 objectives, coordinate-chart structure-preserving L-BFGS-B.
- `intrinsic.py`: genuine product-manifold steepest descent on Skew x SPD (affine-invariant metric) x Euclidean with SPD geodesic update; demonstration solver, not a competitive fast implementation.
- `stress.py`: 15 paired tests of directly optimizing p=32 versus p homotopy 2/4/8/16/32.
- `rankone.py`: small r=1 robustness tests across 24 cases.
- `margin.py`: optional dissipation floor R=LL^T+rho I.
- `tradeoffs.py`: true Hankel/H2 and sampled H-infinity errors.
- `audit.py`: analytic gradients checked with central finite differences; independent trace-power vs HSV norm checks; plotting.

NOTE: The plain `H2` target in `experiment.py` is **squared H2**; the outputs in `tradeoffs.py` display its positive square root.

## Key warnings

1. Direct p=32 was as accurate as homotopy in all 15 paired checks and usually faster.
2. Better Hankel error can worsen H2 and H-infinity errors.
3. The simple relative optimality certificate uses the *unconstrained* HSV lower bound; it is valid but not a new structure-specific inequality.
4. Synthetic pH systems and no direct IRKA-PH / Sato benchmark yet.
