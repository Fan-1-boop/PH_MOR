# Research context / new-chat handoff

## Problem
Find a passive/stable reduced transfer function with the smallest **Hankel-norm error**, initially with strictly proper SISO pH systems and identical feedthrough `D_r=D=0`. An initial convenient parameterization is `A_r=J_r-R_r`, `J_r=-J_r^T`, `R_r\succ0`, `C_r=B_r^T`; general pH systems allow additional energy matrices `Q_r\succ0`.

### Critical conventions
- The Glover/AAK benchmark `sigma_(r+1)(G)` is the unrestricted stable degree-<=r Hankel distance. It is a **lower bound** for any more restrictive passive optimization problem.
- Hankel distance does not detect constant direct feedthrough. Without `D_r=D`, a passive realization may be recovered from a non-passive dynamic approximant by sufficiently large feedthrough, invalidating some passivity-gap arguments.
- Distinguish `degree <= r` from `degree = r` with nonzero coupling. The V2 global minimizer is the zero model for the degree-<=1 class; exact-order problems may have a nonattained infimum.
- For real SISO systems the Hankel operator is self-adjoint, with signed eigenvalues. For general nonsymmetric MIMO systems this is not directly true.

## Phase 1: Schatten objectives and Riemannian optimization
- Compared `H_2`, `S_2` (HSH), and high-order Schatten `S_p` error objectives on small synthetic pH examples.
- Preserved strict pH structure through SPD coordinates and an independent intrinsic SPD-manifold demonstrator.
- High `p` generally improved *true Hankel error*, but sometimes worsened `H_2`.
- Crucial **negative result**: staged `p=2,4,8,16,32` homotopy did not outperform directly optimizing `p=32` in the initial 15 paired trials; the direct version was usually faster. **Do not claim an established continuation advantage.**
- No proven global convergence, no full-scale benchmark against IRKA-PH and structure-preserving BT, and no accepted novel theorem.

## Phase 2: strict passive-versus-unstructured Hankel gap
- Explicit four-state strictly passive SISO pH oscillator example, approximated by degree at most one with `D_r=D=0`.
- Reported spectral data: unrestricted `sigma_2(G) ≈ 4.926627`; passive constrained optimum `≈ 5.069943`; relative gap `≈ 2.909%`.
- A one-pole passive strictly proper approximant has `k/(s+a)` with `k >= 0`, making its Hankel operator PSD. The original signed Hankel spectrum has a negative mode at `-5.069943`, which survives any PSD perturbation; the feasible zero ROM attains that bound.
- This **does not** imply a novel general optimality theorem: it follows from classical spectral variational principles. Computations use floating point; the word "certified" in filenames means a mathematical inequality with numerical estimates, not outward-rounded interval certification.
- Exponential Ritz test spaces gave numerical approximations to the signed spectral lower bound. Frequency-aware shifts helped; large sparse scalability is unproven.
- A simple inertia/rank bound for r>1 may be obtained from classical theory, but is often weaker than the unstructured HSV bound; rigorous nontrivial higher-order improvement remains open in our exploration.

## Target for the next research stage
1. Independently audit V1 and V2 derivations, including Gramian conventions, feedthrough restrictions, numerical claims, and hidden assumptions.
2. Literature audit: Glover/AAK, KYP positive-real lemma, pH/H2 Riemannian MOR, positive-real balancing, passivity-preserving MOR, passive Hankel-approximation results. Separate *known* from *new*.
3. Seek a **nontrivial and computable higher-order (r>=2) structure-aware Hankel lower bound**, ideally significantly sharper than `sigma_(r+1)`, or show why such an improvement is impossible under a proposed generalization.
4. Develop a passive ROM solver with independently computed true Hankel errors, a defensible upper bound and a useful constrained lower bound; benchmark against serious structure-preserving baselines.
5. Only discuss top-journal novelty after a distinct theorem *and* strong numerical evidence exist.

## Reproduce
`pip install -r requirements.txt`
Then run scripts from each phase directory, not project root, because outputs are written relative to script locations.

## Publication status
Exploratory, **not ready for a top-tier journal submission**. No full formal new higher-order theorem and no robust large-scale competitiveness are established.
