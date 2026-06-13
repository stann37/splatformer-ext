#!/bin/bash
# T2.1 soak: 5k-step comparison (baseline vs octree-default) on the 50-scene subset.
# Full objaverse test-set eval at step 0 (input + identity) and at the final step.
# Usage: [GPUS=1,2] bash scripts/soak_train.sh {baseline|octree-default} [steps]
set -e
CONFIG=$1
STEPS=${2:-5000}
GPUS=${GPUS:-1,2}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS
LAST=$((STEPS - 1))

COMMON_ARGS=(
    train.py
    --gin_file=configs/dataset/objaverse.gin
    --gin_file=configs/train/default.gin
    --gin_param="build_trainloader.batch_size=$NPROC"
    --gin_param="build_trainloader.accumulate_step=1"
    --gin_param="build_trainloader.num_workers=2"
    --gin_param="total_steps=$STEPS"
    --gin_param="training.log_interval=20"
    --gin_param="training.save_interval=$STEPS"
    --gin_param="training.eval_interval=$LAST"
    --gin_param="training.log_image_interval=1000000"
    --gin_param="test_dataset/SplatfactoDataset.nerfstudio_folder={'objaverse':'test-set/objaverseOOD/nerfstudio'}"
    --gin_param="test_dataset/SplatfactoDataset.colmap_folder={'objaverse':'test-set/objaverseOOD/colmap'}"
    --wandb_dir=outputs/wandb
)

case $CONFIG in
  baseline)
    OUT=outputs/t2.1_soak_baseline
    ARGS=(--gin_file=configs/model/ptv3.gin)
    ;;
  octree-default)
    OUT=outputs/t2.1_soak_octree-default
    ARGS=(--gin_file=configs/model/octree_ptv3.gin)
    ;;
  *)
    echo "unknown config: $CONFIG"; exit 1
    ;;
esac

mkdir -p outputs/wandb
WANDB_MODE=offline torchrun --nnodes=1 --nproc_per_node=$NPROC --rdzv-endpoint=localhost:29522 \
    "${COMMON_ARGS[@]}" "${ARGS[@]}" --output_dir=$OUT
