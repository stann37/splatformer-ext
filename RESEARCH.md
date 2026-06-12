# SplatFormer extensions — research log

Fork of [ChenYutongTHU/SplatFormer](https://github.com/ChenYutongTHU/SplatFormer) (ICLR'25).
Goal: improve PTV3-based 3DGS refinement for OOD novel view synthesis, with a focus on
**unbounded scenes** and better use of **observable signals**.

## Philosophy (the constraint that shapes everything)

**Reconstruction, not hallucination.**
- Extreme-angle test views still *observe* the scene; the information exists in the
  input views at lower angular density. The model fails because its inductive biases
  waste capacity, not because the problem is underdetermined.
- Therefore: NO diffusion, NO generative priors, NO 2D foundation-model guidance.
- Prefer exposing observable signals (camera poses, frustum visibility, rendered
  images) and fixing spatial discretization (adaptive voxels) over learned priors.

## Selected directions (2026-06-12)

| ID | Direction | One-line rationale | Status |
|----|-----------|--------------------|--------|
| T1.1 | Per-Gaussian visibility features | "seen by test but not train" exactly flags artifact sources; ~80 LOC | branch `t1.1-visibility` |
| T1.3 | Scene contraction (`x/(1+‖x‖)`) | MinMaxScaler crushes foreground in unbounded scenes (~870× resolution loss measured) | branch `t1.3-contraction` |
| T1.4 | Multi-view consistency loss | Photometric loss is per-view; reprojection consistency adds a 3D constraint | branch `t1.4-mv-consist` |
| T2.1 | Octree-PTV3 (adaptive voxels, SVRaster-style) | Uniform 384³ grid wastes capacity in empty space, collides in dense regions (13.9% co-voxel) | branch `t2.1-octree-ptv3` |
| T2.2 | View-conditioned PTV3 (camera tokens, cross-attn) | Test cameras are known at inference but the model never sees them | branch `t2.2-view-cond` |
| T3.1 | Visibility-biased octree | Synthesis of T1.1+T2.1: subdivide deeper where test views look | branch `t3.1-vis-octree` (depends on T1.1, T2.1) |

Deferred (future work, not now): render-aware iterative refinement (T2.3),
MAE pretraining on 3DGS (T3.2), SO(3)-equivariant SH processing (T3.3).

Explicitly rejected: diffusion residuals, SDS/score distillation, 2D foundation priors
(all hallucinate), naive bigger uniform grid (cubic memory, paper says limited returns).

## Decisions made

### [2026-06-12] Adaptive voxels: octree, not hash grid
- Octree (pointer-based flat arrays + torch_scatter) is a *strict generalization* of
  the uniform grid: `max_depth=9, max_pts_per_leaf=1` ≈ current behavior → clean ablation.
- Morton-code DFS order doubles as the space-filling-curve serialization PTV3 needs.
- Hash grid (Instant-NGP style) kept as fallback for positional features only (T1.2-ish).

### [2026-06-12] Octree structure: deterministic, not learnable
- Octree **metadata goes in as features** (leaf depth, intra-leaf position, leaf density,
  boundary indicator — ~8 dims) at both backbone input and output head.
- Subdivision criterion is deterministic but **visibility-biased** (T3.1):
  `importance = point_count + α·visibility_test` — task-aware without learning machinery.
- Rejected fully-differentiable subdivision (Gumbel): tree topology changing under the
  model destroys gradient continuity; density is well-captured by simple criteria.
- Two-stage learned subdivision predictor: interesting, deferred to future work.

### [2026-06-12] Key output-head detail for octree
- Per-Gaussian residual head input = [leaf_feat, original 23-d Gaussian feat,
  **intra-leaf position**]. The intra-leaf position is what distinguishes co-leaf
  Gaussians — without it, adaptive voxels lose per-Gaussian identity.

## Empirical findings so far

### Reproduction (branch `reproduction`, 2026-06-12)
| Test set | Ours | Paper | Baseline 3DGS (ours) |
|----------|------|-------|----------------------|
| Objaverse-OOD (40 scenes) | 23.03 / 0.820 / 0.170 | 23.06 / 0.821 / 0.170 | 19.19 PSNR |
| GSO-OOD (40) | 25.02 / 0.863 / 0.148 | 25.01 / 0.863 / 0.148 | 22.21 |
| Real-OOD (4) | 24.36 / 0.902 / 0.100 | 24.33 / 0.902 / 0.100 | 23.59 |
| ShapeNet-OOD (20) | 27.93 / 0.920 / 0.136 | 27.98 / 0.920 / 0.136 | 21.00 |

(Our "Input 3DGS" baseline differs slightly from paper's table — paper reports 30k-step
3DGS; the released test set ships 10k/20k-step 3DGS. Noted in README upstream.)

### What the model actually does (analyze_splats.py, carrot scene)
- median ‖Δmeans‖ = 4e-4 (Tanh-bounded — barely moves Gaussians)
- mean Δlog(scale) = +0.39 (enlarges splats ~50%)
- mean Δopacity = −0.07 (softens)
- → strategy is "keep positions, soften splats", consistent with removing
  view-overfit artifacts.

### Voxel collision measurement (voxel_collisions.py, carrot scene, G=384)
- 43,971 Gaussians → 40,399 occupied cells (0.07% of 56.6M)
- 13.9% of Gaussians share a cell; max 10 per cell; worse at pooled stages (max 21 at G=192)
- → motivates octree + intra-leaf positional features.

### Paper's own unbounded attempt (Appendix H, Table H.2 — MVImgNet)
- 3DGS 19.81 → SplatFormer 21.68 PSNR (+1.87, vs +3-6 dB object-centric)
- Authors blame "normalization and downpooling … disproportionately downscales
  foreground objects" → exactly what T1.3 + T2.1 attack.
- **Win condition for T2.1+T1.3: beat 21.68 PSNR on their MVImgNet protocol.**
  (Caveat: their 4k-scene training set + 70-scene eval split is not released;
  we must regenerate following Appendix H.)

## Environment quirks (this machine)

- conda env: `splatformer` (Python 3.8, CUDA 11.8 toolkit via conda, PyTorch 2.1.2+cu118)
- **Compiler:** system default gcc-12 lacks g++-12/cc1plus → CUDA extension builds fail.
  Activation hook at `$CONDA_PREFIX/etc/conda/activate.d/compiler_env.sh` exports
  `CC/CXX/CUDAHOSTCXX=/usr/bin/{gcc,g++}-11`. Keep it.
- Stray symlink `$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include/crypt.h`
  from an abandoned conda-cross-compiler attempt — harmless, deletable.
- Packages missing from requirements.txt that are actually needed:
  `gin-config`, `pointops` (build from `Pointcept/libs/pointops/` with the gcc-11 env).
- 3× RTX A4500 (20GB). **Always run eval with `--nproc_per_node=3`** (or
  `CUDA_VISIBLE_DEVICES=0` with 1 proc) — see commit 6402d6c for the sharding bug.
- Data: `test-set/` (1.5GB, from the paper's Google Drive), `checkpoints/`
  (objaverse + shapenet, ~190MB each). Both gitignored; re-download via gdown
  (IDs in upstream README).

## Open questions

- Octree leaf aggregation: mean-pool vs small set-attention — measure both in T2.1.
- T1.4: use rendered depth (re-rasterize with colors=depth) vs modifying gsplat — start
  with re-rasterize.
- MVImgNet training-set regeneration cost (4k scenes × 3DGS training) — budget GPU-weeks?

## T1.1 progress (2026-06-13, branch `t1.1-visibility`)

Implemented + all tests green (commit 30e8548): 7 per-Gaussian frustum-visibility
features (poses only): coverage_tr/te, disagreement, diversity_tr/te, dir_shift,
elev_shift. Computed in `dataset/GS.py:load_scene` in normalized space; camera
convention cross-checked against gsplat rendering.

**Design findings (empirically validated):**
- Frustum coverage saturates (~1.0) on object-centric scenes — wide FoV puts the whole
  object in every frustum and there is no occlusion modeling -> `disagreement` ~ 0
  there. It should become informative on unbounded/real captures (T1.3 + MVImgNet).
- The informative object-centric signals are *angular*: on a real Objaverse-OOD scene,
  div_te=0.03 vs div_tr=0.86 and elev_shift=-0.85 (test views clustered top-down vs
  ringed training views).
- Mean-direction shift degenerates under ring symmetry (resultant collapses);
  elevation shift is the robust complement. Both kept (azimuth-OOD vs elevation-OOD).

Next for T1.1: training comparison (ptv3_visibility.gin vs ptv3.gin) on the train
subset — needs GPUs free of the T2.1 soak. Then T3.1: feed
`0.5*(div_tr - div_te) + |elev_shift|`-style importance into OctreePT's
`importance` input (already plumbed).

## Experiment log

| Date | Branch | Config | PSNR (Obj/GSO/Real/ShapeNet) | Notes |
|------|--------|--------|------------------------------|-------|
| 2026-06-12 | reproduction | paper ckpts, uniform PTV3 | 23.03 / 25.02 / 24.36 / 27.93 | baseline reproduction, matches paper |

## Next session pickup point

> **Start T1.1 (visibility features).** Plan is fully specced in the conversation
> notes: new `utils/visibility.py` (frustum test → 5 per-Gaussian features),
> inject in `dataset/GS.py:__iter__`, add `'visibility'` to
> `FeaturePredictor.input_features` + `FEATURE2CHANNEL`. Then train on Objaverse-OOD
> and compare to the 23.03 baseline. Before training: also need the *training set*
> (train-set/objaverseOOD), which is NOT yet downloaded — the small subset link is in
> the upstream README.
