"""
Adaptive-octree serialization utilities for PTV3 (T2.1).

Core idea (see docs/t2.1-octree-design.md): PTV3's SerializedPooling is uniform-depth
octree pooling (code >> 3 = parent cell). We generalize it to an importance-weighted
adaptive octree: per-point leaf depths d_i, per-stage target depths T_s, tokens pool to
depth t_i = min(d_i, T_s) with depth-tagged group keys.

Degenerate case (leaf_K=1, weights=1, D_max=9, schedule=(9,8,7,6)) reproduces the
baseline exactly.
"""
import math
import torch
import torch.nn as nn
import torch_scatter
from addict import Dict

from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.structure import Point
from pointcept.models.utils.serialization import encode

DEPTH_TAG_BITS = 5  # depth tag in low bits of group keys; supports D_max < 32


@torch.no_grad()
def z_order_encode_simple(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    """Batch-free z-order (Morton) code at `depth` bits per axis. (N,) int64."""
    code = encode(grid_coord, batch=None, depth=depth, order="z")
    return code


@torch.no_grad()
def assign_leaf_depths(
    morton: torch.Tensor,      # (N,) full-depth (D_max) z-order codes, batch-free
    batch: torch.Tensor,       # (N,) batch index
    d_max: int,
    leaf_k: float = 1.0,
    weights: torch.Tensor = None,   # (N,) importance weights; None = uniform 1.0
) -> torch.Tensor:
    """Per-point octree leaf depth: smallest d where the weighted count of the point's
    depth-d cell is <= leaf_k; D_max if never. Monotone in d (counts shrink with depth),
    so one pass over depths suffices. Returns (N,) int64 in [0, d_max]."""
    n = morton.shape[0]
    device = morton.device
    if weights is None:
        weights = torch.ones(n, device=device)
    leaf_depth = torch.full((n,), d_max, dtype=torch.int64, device=device)
    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    for d in range(d_max + 1):
        prefix = morton >> (3 * (d_max - d))
        key = (batch.long() << (3 * d_max + 1)) | prefix
        _, inv = torch.unique(key, return_inverse=True)
        cell_w = torch_scatter.scatter_add(weights, inv, dim=0)
        ok = unassigned & (cell_w[inv] <= leaf_k)
        leaf_depth[ok] = d
        unassigned = unassigned & ~ok
        if not unassigned.any():
            break
    return leaf_depth


@torch.no_grad()
def group_keys(
    morton: torch.Tensor,       # (N,) full-depth codes, batch-free
    batch: torch.Tensor,        # (N,)
    token_depth: torch.Tensor,  # (N,) per-token depth t_i (already min(d_i, T_s))
    d_max: int,
) -> torch.Tensor:
    """Depth-tagged group key: batch | prefix | depth. Tokens merge iff same batch,
    same depth and same prefix at that depth. (N,) int64."""
    shift = 3 * (d_max - token_depth)
    prefix = morton >> shift
    key = (batch.long() << (3 * d_max + DEPTH_TAG_BITS)) | (prefix << DEPTH_TAG_BITS) | token_depth
    return key


@torch.no_grad()
def stage_grid_coord(
    grid_coord_dmax: torch.Tensor,  # (N, 3) int, full-resolution (2^D_max) grid coords
    token_depth: torch.Tensor,      # (N,) t_i <= T_s
    d_max: int,
    target_depth: int,              # T_s
) -> torch.Tensor:
    """Cell center expressed on the 2^{T_s} grid; used for per-stage sort codes and CPE
    sparse-conv coords. For t_i == T_s this is the cell coord itself (degenerate ==
    baseline). Centers are unique across mixed depths because token cells are disjoint.
    Pure bit-masking on grid coords — no Morton decode needed. Returns (N, 3) int32."""
    gc = grid_coord_dmax.long()
    shift = (d_max - token_depth).unsqueeze(-1)                  # (N, 1)
    corner_dmax = (gc >> shift) << shift                          # mask to cell corner at t_i
    corner_ts = corner_dmax >> (d_max - target_depth)             # corner on the T_s grid
    half = torch.where(
        token_depth < target_depth,
        1 << (target_depth - token_depth - 1).clamp(min=0),
        torch.zeros_like(token_depth),
    )
    return (corner_ts + half.unsqueeze(-1)).int()


@torch.no_grad()
def octree_point_features(
    morton: torch.Tensor,        # (N,) full-depth codes
    batch: torch.Tensor,
    coord: torch.Tensor,         # (N, 3) float in [0, 1]
    leaf_depth: torch.Tensor,    # (N,)
    d_max: int,
) -> torch.Tensor:
    """Per-Gaussian octree features (6 dims): normalized leaf depth, intra-leaf
    position (3), log leaf point count, log leaf size."""
    key = group_keys(morton, batch, leaf_depth, d_max)
    _, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    leaf_count = counts[inv].float()
    leaf_size = (0.5 ** leaf_depth.float())                                # side length in [0,1] coords
    cell_corner = (coord / leaf_size.unsqueeze(-1)).floor() * leaf_size.unsqueeze(-1)
    intra = ((coord - cell_corner) / leaf_size.unsqueeze(-1)).clamp(0, 1)
    return torch.cat(
        [
            (leaf_depth.float() / d_max).unsqueeze(-1),
            intra,
            (torch.log1p(leaf_count) / math.log(100.0)).unsqueeze(-1),
            (torch.log(leaf_size + 1e-12) / math.log(2.0) / -d_max).unsqueeze(-1),
        ],
        dim=1,
    )  # (N, 6)


class AdaptiveSerializedPooling(PointModule):
    """Octree-aware replacement for Pointcept's SerializedPooling.

    Pools each token to depth t_i = min(token_depth_i, target_depth) using depth-tagged
    group keys, then re-serializes all orders at the common stage depth via cell-center
    coords. Produces pooling_parent / pooling_inverse identical in spirit to the
    baseline, so SerializedUnpooling works unchanged.

    Expects the incoming Point to carry: octree_code (N,), token_depth (N,), batch,
    feat, coord, and the usual serialized_* fields (for assertion only).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        target_depth,           # T_s for this stage
        d_max,
        orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_depth = int(target_depth)
        self.d_max = int(d_max)
        self.orders = orders
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    def forward(self, point: Point):
        assert {"octree_code", "token_depth", "grid_coord_dmax", "batch", "feat", "coord"}.issubset(point.keys())

        new_depth = torch.minimum(
            point.token_depth,
            torch.full_like(point.token_depth, self.target_depth),
        )
        key = group_keys(point.octree_code, point.batch, new_depth, self.d_max)

        _, cluster, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        # Pooled token attributes. The head is a valid representative: its octree_code /
        # grid_coord_dmax prefix at new_depth defines the cell (lower bits are masked
        # downstream wherever the cell identity matters).
        pooled_code = point.octree_code[head_indices]
        pooled_depth = new_depth[head_indices]
        pooled_batch = point.batch[head_indices]
        pooled_gc_dmax = point.grid_coord_dmax[head_indices]

        grid_coord = stage_grid_coord(pooled_gc_dmax, pooled_depth, self.d_max, self.target_depth)

        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=grid_coord,
            grid_coord_dmax=pooled_gc_dmax,
            octree_code=pooled_code,
            token_depth=pooled_depth,
            batch=pooled_batch,
        )
        point_new = Point(point_dict)
        # Re-serialize all orders at the common stage depth (cell centers interleave
        # coarse and fine tokens correctly along the curve).
        point_new.serialization(order=self.orders, depth=self.target_depth, shuffle_orders=self.shuffle_orders)

        if self.traceable:
            point_new["pooling_inverse"] = cluster
            point_new["pooling_parent"] = point
        if self.norm is not None:
            point_new = self.norm(point_new)
        if self.act is not None:
            point_new = self.act(point_new)
        point_new.sparsify(pad=96)
        return point_new
