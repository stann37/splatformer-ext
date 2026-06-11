"""
Unit tests for models/adaptive_octree.py (T2.1).

Run:  conda run -n splatformer python tests/test_adaptive_octree.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_scatter

from models.adaptive_octree import (
    assign_leaf_depths,
    group_keys,
    stage_grid_coord,
    octree_point_features,
    AdaptiveSerializedPooling,
    z_order_encode_simple,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_cloud(n=20000, d_max=12, seed=0, device=DEVICE, clusters=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if clusters:
        # a few dense clusters + uniform background, mimicking object + floaters
        centers = torch.rand(8, 3, generator=g)
        pts = []
        for c in centers:
            pts.append(c + 0.01 * torch.randn(n // 10, 3, generator=g))
        pts.append(torch.rand(n - len(pts) * (n // 10), 3, generator=g))
        coord = torch.cat(pts).clamp(0, 1 - 1e-6)
    else:
        coord = torch.rand(n, 3, generator=g)
    coord = coord.to(device)
    grid = (coord * (2 ** d_max)).floor().long().clamp(0, 2 ** d_max - 1)
    morton = z_order_encode_simple(grid, depth=d_max).to(device)
    batch = torch.zeros(coord.shape[0], dtype=torch.long, device=device)
    return coord, grid, morton, batch


def test_leaf_depth_monotone_and_correct():
    """A point alone in its depth-d cell must get leaf_depth <= d; points in the same
    cell at D_max must get leaf_depth == D_max (with K=1)."""
    d_max = 6
    # two points in the same finest cell + one isolated point
    grid = torch.tensor([[0, 0, 0], [0, 0, 0], [32, 32, 32]], dtype=torch.long, device=DEVICE)
    morton = z_order_encode_simple(grid, depth=d_max).to(DEVICE)
    batch = torch.zeros(3, dtype=torch.long, device=DEVICE)
    d = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0)
    assert d[0] == d_max and d[1] == d_max, f"co-cell points must hit d_max, got {d}"
    assert d[2] <= 1, f"isolated far point should resolve at depth<=1, got {d[2]}"
    print(f"[ok] leaf depth correctness: {d.tolist()}")


def test_leaf_depth_weighted():
    """With low weights, dense clusters merge early (small leaf depth)."""
    d_max = 10
    coord, grid, morton, batch = make_cloud(n=5000, d_max=d_max, clusters=True)
    d_uniform = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0)
    w = torch.full((coord.shape[0],), 0.01, device=DEVICE)  # everything unimportant
    d_weighted = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0, weights=w)
    assert (d_weighted <= d_uniform).all()
    assert d_weighted.float().mean() < d_uniform.float().mean() - 1.0, (
        f"low weights should shrink depths: {d_weighted.float().mean():.2f} vs {d_uniform.float().mean():.2f}")
    print(f"[ok] weighted depths: uniform mean {d_uniform.float().mean():.2f} -> weighted {d_weighted.float().mean():.2f}")


def test_group_key_no_alias():
    """Tokens at different depths must never share a key; same-cell same-depth must."""
    d_max = 12
    coord, grid, morton, batch = make_cloud(n=20000, d_max=d_max)
    d = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0)
    key = group_keys(morton, batch, d, d_max)
    # same key -> same depth (depth is in the low bits, so this is by construction;
    # verify anyway via reconstruction)
    depth_from_key = key & ((1 << 5) - 1)
    uniq, inv = torch.unique(key, return_inverse=True)
    # all members of a group have equal depth
    dmin = torch_scatter.scatter_min(d, inv, dim=0)[0]
    dmax_ = torch_scatter.scatter_max(d, inv, dim=0)[0]
    assert (dmin == dmax_).all(), "a group contains mixed depths -> key aliasing"
    print(f"[ok] group keys: {uniq.numel()} groups for {coord.shape[0]} pts, no mixed-depth groups")


def test_stage_grid_coord_unique():
    """Cell-center stage coords must be unique across mixed-depth tokens."""
    d_max = 12
    target = 9
    coord, grid, morton, batch = make_cloud(n=30000, d_max=d_max)
    d = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0)
    t = torch.minimum(d, torch.full_like(d, target))
    key = group_keys(morton, batch, t, d_max)
    _, inv = torch.unique(key, return_inverse=True)
    # one representative per token
    n_tok = int(inv.max()) + 1
    head = torch.zeros(n_tok, dtype=torch.long, device=DEVICE)
    head.scatter_(0, inv, torch.arange(inv.shape[0], device=DEVICE))
    gc = stage_grid_coord(grid[head], t[head], d_max, target)
    flat = (gc[:, 0].long() << 40) | (gc[:, 1].long() << 20) | gc[:, 2].long()
    assert torch.unique(flat).numel() == n_tok, (
        f"stage grid coords collide: {torch.unique(flat).numel()} unique vs {n_tok} tokens")
    print(f"[ok] stage grid coords unique: {n_tok} tokens at target depth {target}")


def test_degenerate_equivalence():
    """K=1, D_max=9, pooling 9->8 must produce EXACTLY the baseline's clusters."""
    from pointcept.models.utils.structure import Point
    d_max = 9
    coord, grid, morton, batch = make_cloud(n=20000, d_max=d_max)

    # --- baseline grouping: shift the depth-9 serialized code by 3 bits, unique ---
    code9 = z_order_encode_simple(grid, depth=d_max).to(DEVICE)
    base_key = code9 >> 3
    _, base_cluster = torch.unique(base_key, return_inverse=True)

    # --- ours: degenerate octree (all leaf depths effectively >= 8), target depth 8 ---
    d = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0)
    t = torch.minimum(d, torch.full_like(d, 8))
    ours_key = group_keys(morton, batch, t, d_max)
    _, ours_cluster = torch.unique(ours_key, return_inverse=True)

    # Equivalent partitions iff cluster ids are a relabeling of each other
    pair = base_cluster.long() << 32 | ours_cluster.long()
    n_pairs = torch.unique(pair).numel()
    n_base = torch.unique(base_cluster).numel()
    n_ours = torch.unique(ours_cluster).numel()
    assert n_pairs == n_base == n_ours, (
        f"partitions differ: base {n_base}, ours {n_ours}, joint {n_pairs}")
    print(f"[ok] degenerate equivalence: identical partition ({n_base} clusters)")


def test_pooling_module_forward_backward():
    """AdaptiveSerializedPooling forward+backward on GPU, mixed depths."""
    if DEVICE != "cuda":
        print("[skip] pooling module test needs CUDA (spconv)")
        return
    d_max, target = 12, 9
    coord, grid, morton, batch = make_cloud(n=30000, d_max=d_max)
    d = assign_leaf_depths(morton, batch, d_max, leaf_k=4.0)
    feat = torch.randn(coord.shape[0], 32, device=DEVICE, requires_grad=True)

    from pointcept.models.utils.structure import Point
    point = Point(
        coord=coord, grid_coord=grid.int(), grid_coord_dmax=grid.int(), batch=batch, feat=feat,
        octree_code=morton, token_depth=d,
    )
    point.serialization(order=("z", "z-trans", "hilbert", "hilbert-trans"), depth=d_max, shuffle_orders=False)

    pool = AdaptiveSerializedPooling(
        in_channels=32, out_channels=64, target_depth=target, d_max=d_max,
        norm_layer=torch.nn.BatchNorm1d, act_layer=torch.nn.GELU,
    ).to(DEVICE)
    out = pool(point)
    n_tok = out.feat.shape[0]
    assert (out.token_depth <= target).all()
    assert out.serialized_code.shape[1] == n_tok
    out.feat.sum().backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    print(f"[ok] pooling module: {coord.shape[0]} -> {n_tok} tokens (target depth {target}), grads flow")


def test_octree_features():
    d_max = 10
    coord, grid, morton, batch = make_cloud(n=10000, d_max=d_max)
    d = assign_leaf_depths(morton, batch, d_max, leaf_k=1.0)
    f = octree_point_features(morton, batch, coord, d, d_max)
    assert f.shape == (coord.shape[0], 6)
    assert torch.isfinite(f).all()
    assert (f[:, 1:4] >= 0).all() and (f[:, 1:4] <= 1).all(), "intra-leaf pos out of range"
    print(f"[ok] octree features: shape {tuple(f.shape)}, ranges valid")


if __name__ == "__main__":
    test_leaf_depth_monotone_and_correct()
    test_leaf_depth_weighted()
    test_group_key_no_alias()
    test_stage_grid_coord_unique()
    test_degenerate_equivalence()
    test_octree_features()
    test_pooling_module_forward_backward()
    print("\nAll adaptive-octree tests passed.")
