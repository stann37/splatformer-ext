"""
Synthetic-unbounded stress test (T2.1 usefulness).

Question: on an unbounded scene (foreground object + far background), how much
spatial resolution / network capacity does the *foreground* get under
  (a) the baseline uniform grid (G=384, the thing SerializedPooling uses), vs
  (b) our adaptive octree (uniform leaf weights), vs
  (c) the importance-weighted octree (background downweighted — the T3.1 preview)?

We don't need a trained model for this: the quantity that limits refinement quality
is how finely the foreground is discretized and how many transformer tokens it keeps
through pooling. We measure both directly.

Method: take a real object-centric scene (already a clean foreground), inject
synthetic background Gaussians (a far shell + a ground plane) to emulate an unbounded
capture, MinMax-normalize the combined cloud exactly as dataset/GS.py does, then
compare schemes. Foreground vs background is known by construction (we keep the mask).

Run:
  CUDA_VISIBLE_DEVICES=0 conda run -n splatformer \
      python scripts/unbounded_stress_test.py [--bg_scale 10] [--bg_count_mult 2]
"""
import sys, os, glob, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.transform_utils import MinMaxScaler
from models.adaptive_octree import assign_leaf_depths, z_order_encode_simple, group_keys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G_BASELINE = 384          # uniform grid resolution used by SplatFormer's ptv3.gin
D_MAX = 12                # octree finest depth (octree_ptv3.gin)
LEAF_K = 1.0
DEPTH_SCHEDULE = (12, 9, 8, 7)   # per-stage target depths


def load_foreground(scene_dir):
    ckpt_file = sorted(glob.glob(scene_dir + "/nerfstudio_models/step-*.ckpt"))[-1]
    ckpt = torch.load(ckpt_file, map_location="cpu")
    ckpt = {k.replace("_model.gauss_params.", ""): v for k, v in ckpt.items() if "gauss_params" in k}
    return ckpt["means"].to(DEVICE)          # raw world means (N, 3)


def synth_background(fg_means, bg_scale, bg_count_mult, seed=0):
    """Far shell + ground plane around the foreground, in the SAME world frame."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    center = fg_means.median(0).values.cpu()
    radius = (fg_means.cpu() - center).norm(dim=1).quantile(0.9).item()  # foreground radius
    n_fg = fg_means.shape[0]
    n_bg = int(n_fg * bg_count_mult)

    # spherical shell at bg_scale * foreground radius
    dirs = torch.randn(n_bg // 2, 3, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    shell = center + dirs * (bg_scale * radius) * (0.9 + 0.2 * torch.rand(n_bg // 2, 1, generator=g))

    # ground plane below the object, extending out to the shell
    n_floor = n_bg - shell.shape[0]
    fxy = (torch.rand(n_floor, 2, generator=g) - 0.5) * 2 * bg_scale * radius
    fz = torch.full((n_floor, 1), -bg_scale * radius * 0.5)
    floor = center + torch.cat([fxy, fz], dim=1)

    bg = torch.cat([shell, floor], dim=0).to(fg_means.device)
    return bg, radius


def effective_axis_resolution(coords01, mask, grid_or_depths, kind):
    """How many distinct cells per axis the masked subset occupies."""
    sub = coords01[mask]
    if kind == "uniform":
        cells = torch.floor(sub * grid_or_depths).int()
        uniq = torch.unique(cells, dim=0).shape[0]
    else:  # 'octree' — grid_or_depths is per-point leaf depth
        # express each point's cell at ITS leaf depth, count distinct cells
        depths = grid_or_depths[mask]
        gc = torch.floor(sub * (2 ** D_MAX)).long().clamp(0, 2 ** D_MAX - 1)
        shifted = gc >> (D_MAX - depths).unsqueeze(1)            # corner at leaf depth
        key = shifted[:, 0] * (1 << 26) + shifted[:, 1] * (1 << 13) + shifted[:, 2]
        key = key * 16 + depths                                  # depth-tag
        uniq = torch.unique(key).shape[0]
    # report as effective cells-per-axis = cube-root of occupied cells
    return uniq, uniq ** (1 / 3)


def tokens_after_pooling(morton, batch, leaf_depth, target_depth, mask):
    """Distinct tokens the masked subset collapses to at a given stage depth."""
    td = torch.minimum(leaf_depth, torch.full_like(leaf_depth, target_depth))
    key = group_keys(morton, batch, td, D_MAX)
    return torch.unique(key[mask]).shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--bg_scale", type=float, default=10.0,
                    help="background radius as multiple of foreground radius")
    ap.add_argument("--bg_count_mult", type=float, default=2.0,
                    help="#background Gaussians as multiple of #foreground")
    args = ap.parse_args()

    scene = args.scene or sorted(glob.glob("test-set/objaverseOOD/nerfstudio/*/splatfacto"))[0]
    fg = load_foreground(scene)
    bg, fg_radius = synth_background(fg, args.bg_scale, args.bg_count_mult)
    means = torch.cat([fg, bg], dim=0)
    is_fg = torch.zeros(means.shape[0], dtype=torch.bool, device=DEVICE)
    is_fg[: fg.shape[0]] = True

    print(f"scene: {scene.split('/')[-2]}")
    print(f"foreground: {fg.shape[0]} gs   background: {bg.shape[0]} gs "
          f"(shell+floor at {args.bg_scale}x radius)")

    # --- normalize exactly like dataset/GS.py (MinMax over the COMBINED cloud) ---
    scaler = MinMaxScaler()
    coords01 = scaler.fit_transform(means)
    inb = torch.all((coords01 >= 0) & (coords01 <= 1), dim=1)
    coords01, is_fg = coords01[inb], is_fg[inb]
    fg_extent = (coords01[is_fg].max(0).values - coords01[is_fg].min(0).values)
    print(f"after MinMax: foreground occupies {fg_extent.mean().item()*100:.1f}% of the "
          f"unit box per axis (background pushed the box out {1/fg_extent.mean().item():.1f}x)")

    batch = torch.zeros(coords01.shape[0], dtype=torch.long, device=DEVICE)
    morton = z_order_encode_simple(
        torch.floor(coords01 * (2 ** D_MAX)).long().clamp(0, 2 ** D_MAX - 1), depth=D_MAX)

    # leaf depths: uniform weight, and importance-weighted (bg downweighted 0.05)
    d_uniform = assign_leaf_depths(morton, batch, D_MAX, leaf_k=LEAF_K)
    w = torch.where(is_fg, torch.ones_like(coords01[:, 0]), torch.full_like(coords01[:, 0], 0.05))
    d_weighted = assign_leaf_depths(morton, batch, D_MAX, leaf_k=LEAF_K, weights=w)

    print("\n=== Foreground effective resolution (cells per axis) ===")
    _, ax_uni = effective_axis_resolution(coords01, is_fg, G_BASELINE, "uniform")
    _, ax_oct = effective_axis_resolution(coords01, is_fg, d_uniform, "octree")
    _, ax_octw = effective_axis_resolution(coords01, is_fg, d_weighted, "octree")
    print(f"  baseline uniform G=384      : {ax_uni:6.1f}")
    print(f"  adaptive octree (uniform w) : {ax_oct:6.1f}   ({ax_oct/ax_uni:.1f}x baseline)")
    print(f"  adaptive octree (importance): {ax_octw:6.1f}   ({ax_octw/ax_uni:.1f}x baseline)")
    print(f"  median foreground leaf depth: uniform-w {d_uniform[is_fg].float().median():.0f}, "
          f"importance-w {d_weighted[is_fg].float().median():.0f} (max {D_MAX})")

    print("\n=== Token budget per stage: foreground share of total tokens ===")
    print(f"  {'stage':>5} | {'octree uniform-w':>22} | {'octree importance-w':>22}")
    print(f"  {'depth':>5} | {'fg':>6} {'tot':>7} {'fg%':>5} | {'fg':>6} {'tot':>7} {'fg%':>5}")
    all_mask = torch.ones_like(is_fg)
    for td in DEPTH_SCHEDULE:
        fu = tokens_after_pooling(morton, batch, d_uniform, td, is_fg)
        tu = tokens_after_pooling(morton, batch, d_uniform, td, all_mask)
        fw = tokens_after_pooling(morton, batch, d_weighted, td, is_fg)
        tw = tokens_after_pooling(morton, batch, d_weighted, td, all_mask)
        print(f"  {td:>5} | {fu:>6} {tu:>7} {100*fu/tu:>4.0f}% | {fw:>6} {tw:>7} {100*fw/tw:>4.0f}%")
    print(f"  (foreground Gaussians: {int(is_fg.sum())} of {coords01.shape[0]} total)")

    print("\nInterpretation: baseline crushes the foreground onto a coarse sub-grid; the")
    print("octree subdivides by local density so the foreground keeps fine cells. Importance")
    print("weighting collapses the down-weighted background early, so a larger SHARE of the")
    print("(cheaper) token budget is spent on the foreground deep in the network.")


if __name__ == "__main__":
    main()
