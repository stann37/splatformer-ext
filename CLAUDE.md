# splatformer-ext

Research fork of SplatFormer (ICLR'25) extending PTV3-based 3DGS refinement for
OOD novel view synthesis. **Read `RESEARCH.md` first** — it has the philosophy,
selected directions, decisions, findings, and the "next session pickup point".

## Hard constraints

- **Reconstruction, not hallucination**: no diffusion, no generative priors, no
  2D foundation-model guidance. Use observable signals (visibility, camera poses,
  rendered images) and better spatial discretization instead.
- Don't break the `reproduction` branch — it's the known-good baseline that
  matches the paper's numbers.

## Environment

- conda env `splatformer` — activate with `conda activate splatformer`
  (conda is initialized in ~/.bashrc; env auto-exports CC/CXX/CUDAHOSTCXX=gcc-11/g++-11
  via an activation hook — needed because system gcc-12 lacks cc1plus).
- 3× RTX A4500 (20GB each).
- **Eval must use all GPUs**: `torchrun --nproc_per_node=3` — the test dataset shards
  by `torch.cuda.device_count()` independent of world size; 1 proc on 3 GPUs silently
  evaluates only 1/3 of scenes. Alternative: `CUDA_VISIBLE_DEVICES=0` + 1 proc.
- Run eval: `bash scripts/train-on-objaverse_inference.sh` (results in
  `outputs/objaverse_splatformer/test/eval.log`).
- `gin-config` and `pointops` (from `Pointcept/libs/pointops/`) are required but
  missing from requirements.txt.

## Repo layout (what matters)

- `train.py` — entry point; `main()` at bottom, `training()` and `evaluation()` above.
- `models/feature_predictor.py` — FeaturePredictor: backbone + per-feature MLP residual
  heads. Key configs: `input_features`, `output_features_type='res'`, `zeroinit`,
  `input_feat_to_mlp` (concats raw 23-d features into the head input — add new
  per-Gaussian features here AND in `FEATURE2CHANNEL`).
- `models/pointtransformer_v3.py` — PTV3 wrapper; grid serialization happens in
  FeaturePredictor.forward (`grid_coord = floor(coord * grid_resolution)`, G=384).
- `dataset/GS.py` — SplatfactoDataset; `load_gs_params_fromnerfstudio()` does the
  MinMaxScaler unit-cube normalization (the thing T1.3/T2.1 replace).
- `configs/{dataset,model,train}/*.gin` — gin configs, override with `--gin_param`.
- `scripts/analyze_splats.py`, `scripts/voxel_collisions.py` — analysis tools.

## Branches

- `main` — mirrors upstream (ChenYutongTHU/SplatFormer).
- `reproduction` — baseline: eval fixes + analysis tools. Branch new work off this.
- `t1.1-visibility`, `t1.3-contraction`, `t1.4-mv-consist`, `t2.1-octree-ptv3`,
  `t2.2-view-cond`, `t3.1-vis-octree` — one per research direction (see RESEARCH.md).

## Conventions

- Experiment outputs: `outputs/<direction>_<dataset>_<version>/` with a
  `config_snapshot.gin` (dump `gin.config_str()`) and a short `notes.md`.
- Log every experiment in the table at the bottom of RESEARCH.md, and update the
  "Next session pickup point" before ending a session.
- wandb project `splatformer-ext`, tag runs with the direction id (e.g. `t1.1`).
