"""
OctreePointTransformerV3 (T2.1): the vendored PTV3 with AdaptiveSerializedPooling.

Differences from models/pointtransformer_v3.py:
- `depth_schedule` (one octree target depth per pooling stage) replaces `stride`.
- Serialization at `d_max` (default 12 -> 4096^3 finest cells) instead of
  bit_length(grid_resolution) (9 for G=384).
- Per-point leaf depths from importance-weighted density (leaf_k, optional
  per-point `importance` in the input dict — T3.1 plugs visibility in here).
- Optional 6-dim octree features concatenated to the input features.

Decoder, attention blocks, embedding and unpooling are reused unchanged.
"""
from functools import partial
import torch
import torch.nn as nn
import gin

from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.structure import Point
from pointcept.models.point_transformer_v3 import SerializedUnpooling, Block

from .pointtransformer_v3 import PointSequential_intermediate_output
from .adaptive_octree import (
    AdaptiveSerializedPooling,
    assign_leaf_depths,
    octree_point_features,
    z_order_encode_simple,
)


class OctreePointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        d_max=12,
        leaf_k=1.0,
        depth_schedule=(9, 8, 7, 6),       # T_s per pooling (len == num_stages - 1)
        use_octree_features=True,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(64, 96, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(96, 96, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        turn_off_bn=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.d_max = int(d_max)
        self.leaf_k = float(leaf_k)
        self.depth_schedule = tuple(int(t) for t in depth_schedule)
        self.use_octree_features = use_octree_features

        assert self.num_stages == len(self.depth_schedule) + 1
        assert self.num_stages == len(enc_depths) == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)
        assert self.num_stages == len(dec_depths) + 1 == len(dec_channels) + 1
        assert all(t <= self.d_max for t in self.depth_schedule)
        assert all(a >= b for a, b in zip(self.depth_schedule, self.depth_schedule[1:])), \
            "depth_schedule must be non-increasing"

        bn_layer = nn.Identity if turn_off_bn else partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        feat_channels = in_channels + (6 if use_octree_features else 0)
        self.embedding = PointSequential(
            nn.Linear(feat_channels, enc_channels[0]),
            bn_layer(enc_channels[0]),
            act_layer(),
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    AdaptiveSerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        target_depth=self.depth_schedule[s - 1],
                        d_max=self.d_max,
                        orders=tuple(self.order),
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.enc.add(module=enc, name=f"enc{s}")

        # decoder (identical to baseline — unpooling only needs pooling_parent/inverse)
        dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential_intermediate_output()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]): sum(dec_depths[: s + 1])]
            dec_drop_path_.reverse()
            dec = PointSequential()
            dec.add(
                SerializedUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=enc_channels[s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        """data_dict: coord (N,3) in [0,1], feat (N,C), offset (B,); optional
        importance (N,) for weighted leaf depths (T3.1)."""
        point = Point(
            dict(
                coord=data_dict["coord"],
                feat=data_dict["feat"],
                offset=data_dict["offset"],
            )
        )
        coord = point.coord
        grid_dmax = (
            (coord * (2 ** self.d_max)).floor().long().clamp(0, 2 ** self.d_max - 1)
        )
        morton = z_order_encode_simple(grid_dmax, depth=self.d_max)
        leaf_depth = assign_leaf_depths(
            morton, point.batch, self.d_max, leaf_k=self.leaf_k,
            weights=data_dict.get("importance", None),
        )

        if self.use_octree_features:
            oct_feat = octree_point_features(morton, point.batch, coord, leaf_depth, self.d_max)
            point.feat = torch.cat([point.feat, oct_feat], dim=1)

        point["grid_coord"] = grid_dmax.int()
        point["grid_coord_dmax"] = grid_dmax.int()
        point["octree_code"] = morton
        point["token_depth"] = leaf_depth

        point.serialization(order=self.order, depth=self.d_max, shuffle_orders=self.shuffle_orders)
        point.sparsify(pad=96)
        point = self.embedding(point)
        point = self.enc(point)
        point, multiscale_point = self.dec(point)
        return point


@gin.configurable
class OctreePTV3Model(nn.Module):
    """Gin-facing wrapper, mirrors PointTransformerV3Model in pointtransformer_v3.py."""

    def __init__(
        self,
        in_channels,
        enable_flash=True,
        enc_dim=64,
        output_dim=96,
        turn_off_bn=False,
        d_max=12,
        leaf_k=1.0,
        depth_schedule=(9, 8, 7, 6),
        use_octree_features=True,
        enc_depths=(2, 2, 2, 6, 2),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_depths=(2, 2, 2, 2),
        dec_num_head=(4, 4, 8, 16),
    ):
        super().__init__()
        if output_dim == 64:
            dec_channels = (64, 64, 128, 256)
        elif output_dim == 96:
            dec_channels = (96, 96, 128, 256)
        elif output_dim == 128:
            dec_channels = (128, 128, 256, 256)
        else:
            raise ValueError("Unsupported output_dim")
        if enc_dim == 32:
            enc_channels = (32, 64, 128, 256, 512)
        elif enc_dim == 64:
            enc_channels = (64, 96, 128, 256, 512)
        else:
            raise ValueError("Unsupported enc_dim")
        patch = 1024 if enable_flash else 128
        self.backbone = OctreePointTransformerV3(
            in_channels=in_channels,
            d_max=d_max,
            leaf_k=leaf_k,
            depth_schedule=depth_schedule,
            use_octree_features=use_octree_features,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=(patch,) * len(enc_channels),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=(patch,) * len(dec_channels),
            enable_flash=enable_flash,
            turn_off_bn=turn_off_bn,
            shuffle_orders=True,
            drop_path=0.3,
        )
        self.output_dim = dec_channels[0]

    def forward(self, x):
        return self.backbone(x)
