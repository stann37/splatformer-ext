"""
End-to-end smoke test for the OctreePT backbone inside FeaturePredictor (T2.1).

Builds a realistic fake 3DGS scene (clustered means, log-scales, quats, SH), runs the
full FeaturePredictor forward + backward with the octree gin config, and reports
per-stage token counts vs the uniform baseline.

Run:  conda run -n splatformer python tests/test_octree_ptv3_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gin

DEVICE = "cuda"


def make_fake_gs(n=60000, sh_degree=1, seed=0, device=DEVICE):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # clustered means in [0,1]^3 (object surface-ish) + sparse outliers
    centers = torch.rand(12, 3, generator=g) * 0.6 + 0.2
    pts = [c + 0.02 * torch.randn(n // 16, 3, generator=g) for c in centers]
    pts.append(torch.rand(n - len(pts) * (n // 16), 3, generator=g))
    means = torch.cat(pts).clamp(1e-4, 1 - 1e-4)
    sh_dim = (sh_degree + 1) ** 2 - 1
    gs = {
        'means': means,
        'scales': torch.randn(n, 3, generator=g) * 0.5 - 5.0,   # log-scales
        'opacities': torch.randn(n, 1, generator=g),
        'quats': torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        'features_dc': torch.randn(n, 3, generator=g) * 0.3,
        'features_rest': torch.randn(n, sh_dim * 3, generator=g).reshape(n, sh_dim, 3) * 0.05,
    }
    return {k: v.to(device) for k, v in gs.items()}


def count_stage_tokens(model, model_input):
    """Hook the pooling modules to record token counts per stage."""
    counts = []
    hooks = []
    from models.adaptive_octree import AdaptiveSerializedPooling
    for m in model.modules():
        if isinstance(m, AdaptiveSerializedPooling):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out: counts.append(out.feat.shape[0])))
    out = model(model_input)
    for h in hooks:
        h.remove()
    return out, counts


def main():
    # import first so gin configurables are registered before parsing
    from models.feature_predictor import FeaturePredictor
    import models.octree_ptv3  # noqa: F401  (registers OctreePTV3Model)
    gin.parse_config_files_and_bindings(
        ['configs/model/octree_ptv3.gin'],
        ['FeaturePredictor.resume_ckpt=None'],
    )

    model = FeaturePredictor().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OctreePT FeaturePredictor params: {n_params/1e6:.1f}M")

    gs = make_fake_gs()
    n = gs['means'].shape[0]

    # forward
    out = model(batch_normalized_gs=[gs], batch_scene_idx=[0])
    assert len(out) == 1
    refined = out[0]
    for k, v in refined.items():
        ref_shape = gs[k].shape
        assert v.shape == ref_shape, f"{k}: {v.shape} != {ref_shape}"
        assert torch.isfinite(v).all(), f"{k} has non-finite values"

    # zero-init property: output == input at init (residual heads zero-initialized)
    max_dev = max((refined[k] - gs[k]).abs().max().item() for k in refined)
    print(f"zero-init max |output - input| = {max_dev:.2e} (expect 0)")
    assert max_dev == 0.0, "zeroinit violated: model is not identity at init"

    # backward at zero-init: only the 6 heads' last layers (weight+bias) get grads —
    # the zero weight matrix blocks everything upstream. This is by design.
    loss = sum(v.sum() for v in out[0].values())
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"params with nonzero grad at zero-init: {n_grad} (expect 12 = 6 heads x w+b)")
    assert n_grad == 12, f"expected exactly 12 at zero-init, got {n_grad}"

    # un-block the heads (simulate one optimizer step) and verify full gradient flow
    model.zero_grad()
    with torch.no_grad():
        for head in model.features_outputhead.values():
            head[-1].weight.normal_(0, 1e-3)
    out2 = model(batch_normalized_gs=[gs], batch_scene_idx=[0])
    loss2 = sum(v.sum() for v in out2[0].values())
    loss2.backward()
    n_grad2 = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"params with nonzero grad after unblocking: {n_grad2}/{n_total}")
    assert n_grad2 > 0.8 * n_total, "gradients do not reach most of the backbone"

    # token counts per stage (re-run forward with hooks, no grad)
    with torch.no_grad():
        _, counts = count_stage_tokens(model.backbone.backbone,
                                       {'coord': gs['means'],
                                        'feat': torch.randn(n, model.gs_features_dim, device=DEVICE),
                                        'offset': torch.tensor([n], device=DEVICE)})
    print(f"tokens per pooled stage (N={n}): {counts}")

    # memory
    print(f"peak CUDA memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("\nOctreePT smoke test passed.")


if __name__ == "__main__":
    main()
