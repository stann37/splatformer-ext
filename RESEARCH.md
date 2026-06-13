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

## T2.1 progress (2026-06-12, branch `t2.1-octree-ptv3`)

**Key discovery (paper framing):** PTV3's `SerializedPooling` IS uniform-depth octree
pooling (`code >> 3` = parent cell). Our method is a strict generalization:
importance-weighted adaptive leaf depths + depth-tagged group keys + per-stage target
depths. Full design: `docs/t2.1-octree-design.md`.

Implemented & tested (commits f4d595f, 2832139):
- `models/adaptive_octree.py` — leaf-depth assignment (weighted), group keys,
  stage grid coords (cell centers, bit-masking only — pointcept's z-order *decode* is
  broken upstream, avoided), octree features, `AdaptiveSerializedPooling`.
- `models/octree_ptv3.py` — backbone + `OctreePTV3Model` gin wrapper.
- `FeaturePredictor` `backbone_type='OctreePT'`; `importance` kwarg plumbed (T3.1 hook).
- `configs/model/octree_ptv3.gin` — default d_max=12, leaf_k=1, schedule (12,9,8,7).
- Tests all green: degenerate partition == baseline partition (the strict-generalization
  check); zero-init identity exact at FeaturePredictor level; grads 496/496 after
  unblocking heads (exactly 12 at zero-init — expected, the zero last layers block
  upstream flow); real-scene (44k gs) fwd+bwd peak 12.3 GB.
- Baseline-semantics note: stride=(1,2,2,2) means first "pooling" is a stride-1 dedupe
  at depth 9; degenerate config is therefore d_max=9 + schedule (9,8,7,6).

## T2.1 validation results (2026-06-13)

**Smoke (300 steps, 50-scene subset, 2-scene eval):** baseline / octree-degenerate /
octree-default all train; degenerate tracks baseline within noise (lpips@200 0.069 vs
0.070; psnr@200 30.01 vs 30.08) → strict-generalization holds at the *training* level,
not just unit tests. Cost: octree-default 0.56 s/step vs baseline 0.36 (1.54×).

**Soak (5k steps, 50-scene subset, full 40-scene Objaverse-OOD eval), 3×A4500:**

| Config | PSNR 0→5k | SSIM | LPIPS | ms/step |
|--------|-----------|------|-------|---------|
| baseline (uniform PTV3) | 19.19 → **19.49** | 0.687 | 0.283 | 390 |
| octree-default (d12, K1, sched 12/9/8/7) | 19.19 → **19.46** | 0.686 | 0.286 | 602 (1.54×) |

**Interpretation (important, honest):** at this tiny budget (5k steps / 50 scenes vs
paper's 200k / full set) *neither* model meaningfully refines — both barely move off the
input 3DGS (19.19). So the soak is a **"does it break anything" gate, not a win
condition** — and the octree passes it (trains stably for thousands of steps, equal
quality at equal budget, predicted 1.54× cost). The octree's structural advantage
(capacity by density) is on **(a)** fully-trained fine detail and **(b)** unbounded
scenes where the uniform grid *structurally* fails — and (b) is gated on T1.3
(contraction). **Bounded object-centric scenes at low budget are exactly where uniform
PTV3 already suffices, so they cannot demonstrate T2.1's value.** Conclusion: T2.1 is
implemented & proven-correct; its *value demonstration* must come on MVImgNet (needs
T1.3) or a full-length Objaverse run (needs the full train set + GPU-days).

A crash mid-soak (spconv `implicit_gemm` TensorStorage error at step ~4900, transient —
likely a fragmentation/contention hiccup) was fixed by rerunning with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on all 3 GPUs.

## T2.1 usefulness on unbounded scenes (2026-06-13, synthetic stress test)

`scripts/unbounded_stress_test.py`: real foreground object + injected background
(far shell + ground plane), MinMax-normalized exactly as the pipeline does, then we
measure how much *foreground* spatial resolution / token budget survives. No trained
model needed — this isolates the structural variable the octree targets.

Real Objaverse scene (44k fg gs) + synthetic background:

| bg severity | fg % of box | baseline G=384 fg cells/axis | octree fg cells/axis | gain |
|-------------|-------------|------------------------------|----------------------|------|
| 10x radius, 2x count | 6.2% | 14.0 | 33.6 | **2.4x** |
| 20x radius, 3x count | 3.1% | 8.5  | 31.1 | **3.6x** |

The worse the unbounded background, the bigger the octree's advantage — exactly the
regime where the paper's own MVImgNet attempt struggled.

**Importance weighting (T3.1 preview):** downweighting background (w=0.05) lifts the
foreground's *share* of the token budget at the input stage from **19% -> 62%** (bg 20x);
background tokens collapse 161841 -> 48709. Same fg tokens, far cheaper, higher fg share.

**Design finding that motivates T1.3:** the *absolute* depth schedule (12,9,8,7) still
over-pools the foreground at deep stages on unbounded scenes (at 3% box occupancy, the
fg holds only ~1164 tokens by stage-depth 9). The fix is contraction (T1.3): re-center
+ scale so the foreground fills the unit ball, making the absolute schedule appropriate
again. So octree (resolution) and contraction (placement) are complementary, not
redundant — octree alone recovers input-stage resolution; contraction is needed for the
deeper stages to keep foreground capacity. This is the concrete case for doing T1.3 next.

## Experiment log

| Date | Branch | Config | PSNR (Obj/GSO/Real/ShapeNet) | Notes |
|------|--------|--------|------------------------------|-------|
| 2026-06-12 | reproduction | paper ckpts, uniform PTV3 | 23.03 / 25.02 / 24.36 / 27.93 | baseline reproduction, matches paper |
| 2026-06-13 | t2.1 | uniform PTV3, 5k/50-scene | 19.49 (Obj only) | tiny-budget soak gate |
| 2026-06-13 | t2.1 | octree-default, 5k/50-scene | 19.46 (Obj only) | == baseline at equal budget; correctness gate passed |

## Branch status

- **t2.1-octree-ptv3: feature-complete & validated-correct.** Remaining "win" runs
  (full-length Objaverse; MVImgNet) are blocked on data/compute, not code.
- **t1.1-visibility: feature-complete & tested** (7 visibility features, all tests green).
  Training comparison pending free GPUs.

## Next session pickup point

> **T1.3 (scene contraction)** — the highest-value unlock: it's the gate for showing
> T2.1's real advantage on unbounded scenes. Key design constraint discovered: contraction
> must be a **model-input-only** transform. gsplat rendering assumes Euclidean geometry,
> so the rasterizer must keep operating in the similarity-normalized (MinMaxScaler) space;
> contraction applies only to `coord`/`grid_coord` fed to the backbone, NOT to the means
> used for rendering. Implement as a separate `coord_model` path in feature_predictor /
> octree_ptv3, leaving `gs['means']` (render) untouched. Then T3.1 (visibility-weighted
> octree importance — kwarg already plumbed) and the MVImgNet protocol (Appendix H).
> Win condition still: beat 21.68 PSNR on MVImgNet.
