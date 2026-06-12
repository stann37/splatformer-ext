"""
End-to-end T1.1 smoke: real scene through SplatfactoDataset (visibility computed in
load_scene) -> FeaturePredictor forward with the ptv3_visibility config.

Run:  CUDA_VISIBLE_DEVICES=0 conda run -n splatformer python tests/test_visibility_e2e.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gin

from models.feature_predictor import FeaturePredictor
import dataset.Loader  # noqa: F401 — registers SplatfactoDataset/build_testloader with gin
from utils.visibility import VISIBILITY_DIM


def main():
    gin.parse_config_files_and_bindings(
        [
            'configs/dataset/objaverse.gin',
            'configs/model/ptv3_visibility.gin',
        ],
        [
            "test_dataset/SplatfactoDataset.nerfstudio_folder={'objaverse':'test-set-smoke/objaverseOOD/nerfstudio'}",
            "test_dataset/SplatfactoDataset.colmap_folder={'objaverse':'test-set-smoke/objaverseOOD/colmap'}",
            'FeaturePredictor.resume_ckpt=None',
        ],
    )
    from dataset.Loader import build_testloader

    loaders = build_testloader()
    batch = next(iter(loaders['objaverse']))
    gs = batch[0]['gs_params']
    assert 'visibility' in gs, "dataset did not attach visibility features"
    vis = gs['visibility']
    n = gs['means'].shape[0]
    assert vis.shape == (n, VISIBILITY_DIM)
    assert torch.isfinite(vis).all()

    print(f"scene: {batch[0]['scene_name']}  N: {n}")
    names = ['cov_tr', 'cov_te', 'disagree', 'div_tr', 'div_te', 'dir_shift', 'elev_shift']
    for i, nm in enumerate(names):
        col = vis[:, i]
        print(f"  {nm:10s} mean={col.mean():+.3f}  p5={col.quantile(0.05):+.3f}  p95={col.quantile(0.95):+.3f}")

    model = FeaturePredictor().cuda()
    gs_cuda = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in gs.items()}
    out = model(batch_normalized_gs=[gs_cuda], batch_scene_idx=[0])
    dev = max((out[0][k] - gs_cuda[k]).abs().max().item() for k in out[0])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"forward OK ({n_params:.1f}M params), zero-init dev={dev:.2e}")
    assert dev == 0.0
    print("\nT1.1 end-to-end smoke passed.")


if __name__ == '__main__':
    main()
