#!/bin/bash
# T2.1 smoke training: 300 optimizer steps on the 50-scene train subset.
# Usage: [GPUS=0,2] bash scripts/smoke_train.sh {baseline|octree-degenerate|octree-default} [steps]
# GPUS defaults to 0,1,2; on a shared machine pass the free ones explicitly.
# Tiny 2-scene test set keeps the mandatory step-0 eval fast.
set -e
CONFIG=$1
STEPS=${2:-300}
GPUS=${GPUS:-0,1,2}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS

COMMON_ARGS=(
    train.py
    --gin_file=configs/dataset/objaverse.gin
    --gin_file=configs/train/default.gin
    --gin_param="build_trainloader.batch_size=$NPROC"
    --gin_param="build_trainloader.accumulate_step=1"
    --gin_param="build_trainloader.num_workers=2"
    --gin_param="total_steps=$STEPS"
    --gin_param="training.log_interval=5"
    --gin_param="training.save_interval=1000000"
    --gin_param="training.eval_interval=1000000"
    --gin_param="training.log_image_interval=1000000"
    --gin_param="test_dataset/SplatfactoDataset.nerfstudio_folder={'objaverse':'test-set-smoke/objaverseOOD/nerfstudio'}"
    --gin_param="test_dataset/SplatfactoDataset.colmap_folder={'objaverse':'test-set-smoke/objaverseOOD/colmap'}"
    --wandb_dir=outputs/wandb
)

case $CONFIG in
  baseline)
    OUT=outputs/t2.1_smoke_baseline
    ARGS=(--gin_file=configs/model/ptv3.gin)
    ;;
  octree-degenerate)
    OUT=outputs/t2.1_smoke_octree-degenerate
    ARGS=(
      --gin_file=configs/model/octree_ptv3.gin
      --gin_param="OctreePTV3Model.d_max=9"
      --gin_param="OctreePTV3Model.depth_schedule=(9,8,7,6)"
      --gin_param="OctreePTV3Model.use_octree_features=False"
    )
    ;;
  octree-default)
    OUT=outputs/t2.1_smoke_octree-default
    ARGS=(--gin_file=configs/model/octree_ptv3.gin)
    ;;
  *)
    echo "unknown config: $CONFIG"; exit 1
    ;;
esac

mkdir -p outputs/wandb
WANDB_MODE=offline torchrun --nnodes=1 --nproc_per_node=$NPROC --rdzv-endpoint=localhost:29521 \
    "${COMMON_ARGS[@]}" "${ARGS[@]}" --output_dir=$OUT
