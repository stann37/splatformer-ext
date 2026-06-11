"""
Compare distributions of input 3DGS vs SplatFormer-refined 3DGS for one scene.

Usage:
  python scripts/analyze_splats.py <scene_dir>
  python scripts/analyze_splats.py outputs/objaverse_splatformer/test/objaverse/viewer/0a604c1ee9b245c7b2d797a910e53219-10

Writes <scene_dir>/analysis.png and prints summary stats.
"""
import sys, os, math
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt


def load_3dgs_ply(path):
    """Load a 3DGS PLY in the standard SuperSplat / nerfstudio format."""
    ply = PlyData.read(path)
    v = ply['vertex']
    n = len(v)

    means = np.stack([v['x'], v['y'], v['z']], axis=1)             # (N, 3)
    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)   # log-scale
    rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)
    opacities_logit = v['opacity']                                  # pre-sigmoid
    dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)  # SH degree-0 color
    # base color = sigmoid(dc * C0 + 0.5)?  No — per gs_utils.SH2RGB:  rgb = dc*C0 + 0.5
    C0 = 0.28209479177387814
    base_rgb = np.clip(dc * C0 + 0.5, 0, 1)

    return dict(
        means=means,
        scales_log=scales,
        scales_lin=np.exp(scales),
        rots=rots,
        opacities=1.0 / (1.0 + np.exp(-opacities_logit)),  # sigmoid
        base_rgb=base_rgb,
        n=n,
    )


def summarize(label, gs):
    print(f"\n=== {label} ===")
    print(f"  #Gaussians       : {gs['n']}")
    print(f"  means range x    : [{gs['means'][:,0].min(): .4f}, {gs['means'][:,0].max(): .4f}]")
    print(f"  means range y    : [{gs['means'][:,1].min(): .4f}, {gs['means'][:,1].max(): .4f}]")
    print(f"  means range z    : [{gs['means'][:,2].min(): .4f}, {gs['means'][:,2].max(): .4f}]")
    print(f"  scale (linear)   : median={np.median(gs['scales_lin']): .5f}  "
          f"mean={gs['scales_lin'].mean(): .5f}  "
          f"p95={np.percentile(gs['scales_lin'], 95): .5f}")
    print(f"  opacity          : mean={gs['opacities'].mean():.3f}  "
          f"p10={np.percentile(gs['opacities'],10):.3f}  "
          f"p90={np.percentile(gs['opacities'],90):.3f}")
    print(f"  base color (rgb) : mean=({gs['base_rgb'].mean(0)[0]:.3f}, "
          f"{gs['base_rgb'].mean(0)[1]:.3f}, {gs['base_rgb'].mean(0)[2]:.3f})")


def plot(scene_dir, input_gs, refined_gs):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(os.path.basename(scene_dir), fontsize=14)

    # --- Row 1: spatial distribution ---
    for i, axis in enumerate('xyz'):
        ax = axes[0, i]
        ax.hist(input_gs['means'][:, i], bins=60, alpha=0.5, label='Input 3DGS', color='C0')
        ax.hist(refined_gs['means'][:, i], bins=60, alpha=0.5, label='SplatFormer', color='C1')
        ax.set_title(f"means.{axis} distribution")
        ax.set_xlabel(axis); ax.set_ylabel("count"); ax.legend()

    # 3D scatter projected to x-y
    ax = axes[0, 3]
    sub = np.random.choice(input_gs['n'], min(5000, input_gs['n']), replace=False)
    ax.scatter(input_gs['means'][sub, 0], input_gs['means'][sub, 1],
               s=0.5, alpha=0.3, color='C0', label='Input')
    ax.scatter(refined_gs['means'][sub, 0], refined_gs['means'][sub, 1],
               s=0.5, alpha=0.3, color='C1', label='Refined')
    ax.set_title("means (x-y projection, 5k sample)"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal'); ax.legend(markerscale=10)

    # --- Row 2: scales (linear & log) + per-Gaussian shift ---
    ax = axes[1, 0]
    ax.hist(np.log(input_gs['scales_lin']).flatten(), bins=60, alpha=0.5, label='Input', color='C0')
    ax.hist(np.log(refined_gs['scales_lin']).flatten(), bins=60, alpha=0.5, label='Refined', color='C1')
    ax.set_title("log(scale)  (all 3 axes flattened)")
    ax.set_xlabel("log scale"); ax.set_ylabel("count"); ax.legend()

    ax = axes[1, 1]
    # max axis scale per Gaussian
    ax.hist(input_gs['scales_lin'].max(1), bins=60, alpha=0.5, label='Input', color='C0', range=(0, 0.05))
    ax.hist(refined_gs['scales_lin'].max(1), bins=60, alpha=0.5, label='Refined', color='C1', range=(0, 0.05))
    ax.set_title("max-axis scale per Gaussian (clipped at 0.05)")
    ax.set_xlabel("max scale"); ax.set_ylabel("count"); ax.legend()

    ax = axes[1, 2]
    ax.hist(input_gs['opacities'], bins=60, alpha=0.5, label='Input', color='C0')
    ax.hist(refined_gs['opacities'], bins=60, alpha=0.5, label='Refined', color='C1')
    ax.set_title("opacity (sigmoid)")
    ax.set_xlabel("opacity"); ax.set_ylabel("count"); ax.legend()

    # Per-Gaussian position shift (only meaningful if order is preserved)
    ax = axes[1, 3]
    if input_gs['n'] == refined_gs['n']:
        shift = np.linalg.norm(refined_gs['means'] - input_gs['means'], axis=1)
        ax.hist(shift, bins=60, color='C2')
        ax.set_title(f"per-Gaussian position shift (||Δmeans||)\n"
                     f"median={np.median(shift):.4f}, p95={np.percentile(shift,95):.4f}")
        ax.set_xlabel("shift"); ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "N changed,\ncan't pair", ha='center', va='center', transform=ax.transAxes)

    # --- Row 3: colors ---
    for i, c in enumerate('rgb'):
        ax = axes[2, i]
        ax.hist(input_gs['base_rgb'][:, i], bins=60, alpha=0.5, label='Input', color='C0')
        ax.hist(refined_gs['base_rgb'][:, i], bins=60, alpha=0.5, label='Refined', color='C1')
        ax.set_title(f"base color {c.upper()}")
        ax.set_xlabel(c); ax.set_ylabel("count"); ax.legend()

    # Per-Gaussian color shift
    ax = axes[2, 3]
    if input_gs['n'] == refined_gs['n']:
        color_shift = np.linalg.norm(refined_gs['base_rgb'] - input_gs['base_rgb'], axis=1)
        ax.hist(color_shift, bins=60, color='C2')
        ax.set_title(f"per-Gaussian base-color shift\n"
                     f"median={np.median(color_shift):.4f}, p95={np.percentile(color_shift,95):.4f}")
        ax.set_xlabel("||Δrgb||"); ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "N changed", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(scene_dir, 'analysis.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f"\nWrote {out}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    scene_dir = sys.argv[1].rstrip('/')

    in_ply = os.path.join(scene_dir, "point_cloud/iteration_0/point_cloud.ply")
    out_ply = os.path.join(scene_dir, "point_cloud/iteration_1/point_cloud.ply")
    assert os.path.exists(in_ply), f"missing {in_ply}"
    assert os.path.exists(out_ply), f"missing {out_ply}"

    print(f"Loading {in_ply}")
    inp = load_3dgs_ply(in_ply)
    print(f"Loading {out_ply}")
    ref = load_3dgs_ply(out_ply)

    summarize("Input 3DGS", inp)
    summarize("SplatFormer Refined", ref)

    # Useful aggregated comparison
    print("\n=== Aggregated diff ===")
    if inp['n'] == ref['n']:
        d_means = np.linalg.norm(ref['means'] - inp['means'], axis=1)
        d_color = np.linalg.norm(ref['base_rgb'] - inp['base_rgb'], axis=1)
        d_opac = ref['opacities'] - inp['opacities']
        d_scale = np.log(ref['scales_lin']).mean(1) - np.log(inp['scales_lin']).mean(1)
        print(f"  ||Δmeans||  : median={np.median(d_means):.4f}  p95={np.percentile(d_means,95):.4f}")
        print(f"  ||Δrgb||    : median={np.median(d_color):.4f}  p95={np.percentile(d_color,95):.4f}")
        print(f"  Δopacity    : median={np.median(d_opac):+.4f}  mean={d_opac.mean():+.4f}")
        print(f"  Δlog(scale) : median={np.median(d_scale):+.4f}  mean={d_scale.mean():+.4f}")

    plot(scene_dir, inp, ref)


if __name__ == '__main__':
    main()
