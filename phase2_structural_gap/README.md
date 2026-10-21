# Passive Hankel Structure V2

This self-contained numerical study constructs a 4-state passive port-Hamiltonian system for which any strictly proper passive realization of McMillan degree <= 1 has a Hankel error 2.909% higher than the unconstrained Glover lower bound.

It verifies the signed spectrum by a classical SISO cross Gramian, an independent controllability/observability HSV computation, and Ritz compression of the Hankel operator with complex exponentials. It compares unconstrained and passive one-pole approximants with repeated deterministic differential-evolution optimization, and provides parameter robustness evidence.

**Reproduce:** `OPENBLAS_NUM_THREADS=1 python reproduce.py` (numpy, scipy, matplotlib).

**Mathematical caveat:** The strict gap requires **equal zero feedthrough D=D_r=0**. If the approximant may change D, its negative-residue dynamics can be made positive real by a sufficiently large direct term, without changing the Hankel norm. Hence this is not a claim that *passivity alone* always makes Hankel approximation suboptimal.

**Novelty caveat:** The lower bound is a direct application of standard spectral/PSD rank-one approximation principles; it is not being claimed as a previously unknown theorem nor a finished top-journal algorithm. The calculations are double-precision rather than formal interval certification.

See `研究报告_第二阶段.md` for definitions, proof, data, limitations, and next steps.
