# PH_MOR — Structure-Preserving Hankel Model Order Reduction

Research repository for **passive port-Hamiltonian (pH) model order reduction** with Hankel-error objectives, Schatten-`p` surrogates, Riemannian optimization, and structure-aware lower bounds.

**Status: exploratory research, not a published or globally certified algorithm.** The numerical results and proof sketches require independent review before publication.

## Research question

Given a stable, strictly proper, passive system `G`, find a reduced model `G_r` preserving passivity and the original direct feedthrough `D_r=D`, while minimizing the Hankel error:

```text
minimize    || G - G_r ||_H
subject to  order(G_r) <= r; G_r passive; D_r = D
```

The classical Glover/AAK lower bound `sigma_(r+1)(G)` applies without the additional passive-ROM constraint. With strict structural constraints, the achievable minimum may be higher.

## Repository contents

- [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) — key findings, pitfalls, precise assumptions, and next research tasks.
- [phase1_schatten/](phase1_schatten/) — low-dimensional Schatten-`p`, `H_2`, and Riemannian prototype.
- [phase2_structural_gap/](phase2_structural_gap/) — strictly proper passive SISO order-`<=1` example with a signed Hankel spectral lower bound.
- [requirements.txt](requirements.txt) — Python dependencies.

## Quick start

```bash
python -m pip install -r requirements.txt
cd phase2_structural_gap
OPENBLAS_NUM_THREADS=1 python reproduce.py
# Alternative starting point:
cd ../phase1_schatten
OPENBLAS_NUM_THREADS=1 python experiment.py
```

The scripts were generated during exploratory research. Some tests are computationally intensive and update local JSON/PNG files. Re-run and verify results before citing them.

## Important limitations

1. The passive SISO `r <= 1` spectral gap uses **fixed zero feedthrough**. Free feedthrough can invalidate the claimed gap.
2. The admissible approximation class is **degree at most** `r`; the zero transfer function is a valid candidate. Exact-degree requirements may have only an infimum.
3. The signed Hankel spectrum argument is specific to **real symmetric SISO Hankel operators**, not generic MIMO systems.
4. The reported small-scale computations are floating-point checks, not validated interval-arithmetic certificates.
5. Direct optimization of a large Schatten-`p` value performed comparably to staged homotopy in the initial paired trials. A distinct homotopy advantage has **not** been established.
6. A generic pH coordinate parameterization can stay passive during optimization, but this does not guarantee global optimality.

## Research handoff

Open [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) first, then inspect the source and existing negative results. Prioritize nontrivial higher-order structure-specific Hankel bounds and independent benchmarks. Do not represent classical spectral or KYP facts as new theorems.

## Attribution and license

Compiled from the user's prior ChatGPT research-session experiment artifacts. No open-source license is assigned yet; repository visibility alone does not grant a reuse license.
