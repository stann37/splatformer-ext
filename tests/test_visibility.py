"""
Unit tests for utils/visibility.py (T1.1).

Run:  conda run -n splatformer python tests/test_visibility.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

from utils.visibility import frustum_visibility, compute_visibility_features, VISIBILITY_DIM


def look_at_c2w(cam_pos, target=(0.5, 0.5, 0.5)):
    """OpenGL-convention camera-to-world looking at `target` (camera -z = view dir)."""
    cam_pos = torch.tensor(cam_pos, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)
    forward = (target - cam_pos)
    forward = forward / forward.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    if forward.abs()[2] > 0.99:
        up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.cross(forward, up); right = right / right.norm()
    true_up = torch.cross(right, forward)
    c2w = torch.eye(4)
    # OpenGL: x=right, y=up, z=backward
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos
    return c2w


INTR = dict(fx=300.0, fy=300.0, cx=200.0, cy=200.0, width=400, height=400)


def test_in_front_vs_behind():
    """A point centered in view must be visible; a point behind the camera must not."""
    cam = look_at_c2w((0.5, 0.5, 2.0))          # above center, looking down at it
    means = torch.tensor([
        [0.5, 0.5, 0.5],    # in front, on the optical axis
        [0.5, 0.5, 3.0],    # behind the camera
    ])
    vis = frustum_visibility(means, cam.unsqueeze(0), **INTR)
    assert vis[0, 0].item() is True or vis[0, 0].item() == 1
    assert not vis[1, 0]
    print("[ok] front/behind classification")


def test_fov_boundary():
    """Points outside the field of view must be invisible.
    fx=300, W=400 -> half-FoV = atan(200/300) ~= 33.7 deg."""
    cam = look_at_c2w((0.5, 0.5, 1.5), target=(0.5, 0.5, 0.5))
    half_fov = math.atan((INTR['width'] / 2) / INTR['fx'])
    dist = 1.0   # depth of test points below the camera
    r_in = 0.9 * dist * math.tan(half_fov)
    r_out = 1.2 * dist * math.tan(half_fov)
    means = torch.tensor([
        [0.5 + r_in, 0.5, 0.5],
        [0.5 + r_out, 0.5, 0.5],
    ])
    vis = frustum_visibility(means, cam.unsqueeze(0), **INTR)
    assert vis[0, 0], "point inside FoV cone should be visible"
    assert not vis[1, 0], "point outside FoV cone should be invisible"
    print("[ok] FoV boundary")


def test_disagreement_signal_narrow_fov():
    """With a narrow FoV (where frustum coverage can actually discriminate),
    Gaussians under the object get positive disagreement when train views are
    top-down and test views are low-elevation."""
    NARROW = dict(fx=2000.0, fy=2000.0, cx=200.0, cy=200.0, width=400, height=400)
    # half-FoV = atan(200/2000) ~= 5.7 deg — narrow enough that "aimed at the center"
    # and "aimed at the under-region" cones don't overlap (frustum test has no
    # occlusion, so this is the only way coverage can discriminate geometrically)
    train_cams, test_cams = [], []
    under = (0.5, 0.5, 0.1)
    for az in torch.linspace(0, 2 * math.pi, 8)[:-1]:
        r = 1.0
        # train: high elevation, aimed at object center
        train_cams.append(look_at_c2w((0.5 + r * math.cos(az), 0.5 + r * math.sin(az), 1.6)))
        # test: low elevation, aimed at the under-region
        test_cams.append(look_at_c2w((0.5 + r * math.cos(az), 0.5 + r * math.sin(az), 0.35),
                                     target=under))
    train_cams = torch.stack(train_cams)
    test_cams = torch.stack(test_cams)

    means = torch.tensor([
        [0.5, 0.5, 0.5],     # object center: aimed at by train views only
        [under[0], under[1], under[2]],  # under the object: aimed at by test views only
    ])
    f = compute_visibility_features(means, train_cams, test_cams, **NARROW)
    assert f.shape == (2, VISIBILITY_DIM)
    cov_tr, cov_te, disagree = f[:, 0], f[:, 1], f[:, 2]
    print(f"    center: cov_tr={cov_tr[0]:.2f} cov_te={cov_te[0]:.2f} disagree={disagree[0]:+.2f}")
    print(f"    under : cov_tr={cov_tr[1]:.2f} cov_te={cov_te[1]:.2f} disagree={disagree[1]:+.2f}")
    assert cov_tr[0] > 0.8, "object center should be visible from train ring"
    assert disagree[1] > disagree[0] + 0.3, "under-object point must have higher disagreement"
    print("[ok] disagreement signal (narrow FoV)")


def test_direction_and_elevation_shift():
    """Elevation shift must catch ring->topdown OOD (where mean-direction collapses
    under ring symmetry); direction shift must catch azimuth-half splits
    (MVImgNet-style) where elevation is unchanged."""
    ring, topdown, front_half, back_half = [], [], [], []
    for i, az in enumerate(torch.linspace(0, 2 * math.pi, 9)[:-1]):
        r = 1.2
        cam = look_at_c2w((0.5 + r * math.cos(az), 0.5 + r * math.sin(az), 0.55))
        ring.append(cam)
        (front_half if math.cos(az) > 0 else back_half).append(cam)
        topdown.append(look_at_c2w((0.5, 0.5, 1.9)))
    ring = torch.stack(ring); topdown = torch.stack(topdown)
    front_half = torch.stack(front_half); back_half = torch.stack(back_half)

    means = torch.tensor([[0.5, 0.5, 0.5]])

    # Objaverse-style: low ring -> top-down. Ring resultant collapses, so dir_shift
    # is unreliable; elevation shift must fire.
    f = compute_visibility_features(means, ring, topdown, **INTR)
    elev_shift = f[0, 6]
    print(f"    ring->topdown: elev_shift={elev_shift:+.3f}")
    assert elev_shift < -0.5, "topdown test views look much more downward than ring"

    # MVImgNet-style: front half -> back half. Elevation identical; dir_shift fires.
    f2 = compute_visibility_features(means, front_half, back_half, **INTR)
    dir_shift2, elev_shift2 = f2[0, 5], f2[0, 6]
    print(f"    front->back: dir_shift={dir_shift2:.3f}, elev_shift={elev_shift2:+.3f}")
    assert dir_shift2 > 1.0, "opposite azimuth halves must give large direction shift"
    assert abs(elev_shift2) < 0.1, "azimuth split should not change elevation"

    # identical sets -> both shifts ~ 0
    f3 = compute_visibility_features(means, ring, ring, **INTR)
    assert f3[0, 5] < 1e-4 and f3[0, 6].abs() < 1e-4
    print("[ok] direction + elevation shift")


def test_diversity():
    """A point ringed by cameras has high angular diversity; a point seen from one
    side has low diversity."""
    cams = []
    for az in torch.linspace(0, 2 * math.pi, 9)[:-1]:
        cams.append(look_at_c2w((0.5 + 1.2 * math.cos(az), 0.5 + 1.2 * math.sin(az), 0.5)))
    cams = torch.stack(cams)
    one_side = cams[:2]   # only two nearby cameras

    means = torch.tensor([[0.5, 0.5, 0.5]])
    f_ring = compute_visibility_features(means, cams, cams, **INTR)
    f_side = compute_visibility_features(means, one_side, one_side, **INTR)
    div_ring, div_side = f_ring[0, 3], f_side[0, 3]
    print(f"    diversity ring={div_ring:.3f}, one-side={div_side:.3f}")
    assert div_ring > 0.8, "full ring should give high diversity"
    assert div_side < 0.2, "single-side viewing should give low diversity"
    print("[ok] angular diversity")


def test_matches_rasterizer_convention():
    """A Gaussian rendered with nonzero alpha by gsplat must be flagged visible."""
    if not torch.cuda.is_available():
        print("[skip] rasterizer cross-check needs CUDA")
        return
    from utils import gs_utils
    device = 'cuda'
    n = 200
    g = torch.Generator().manual_seed(0)
    means = (0.45 + 0.1 * torch.rand(n, 3, generator=g)).to(device)
    gs = {
        'means': means,
        'scales': torch.full((n, 3), -6.0, device=device),
        'quats': torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device),
        'opacities': torch.full((n, 1), 4.0, device=device),
        'features_dc': torch.rand(n, 3, generator=g).to(device),
    }
    cam = look_at_c2w((0.5, 0.5, 1.8)).to(device)
    rgb, alpha = gs_utils.rasterize_gaussians_to_singleimg(
        gs, cam[:3, :], cx=torch.tensor(INTR['cx']), cy=torch.tensor(INTR['cy']),
        fx=torch.tensor(INTR['fx']), fy=torch.tensor(INTR['fy']),
        width=torch.tensor(INTR['width']), height=torch.tensor(INTR['height']),
        background_color=torch.zeros(3, device=device))
    assert alpha.max() > 0.5, "setup error: nothing rendered"
    vis = frustum_visibility(means, cam.unsqueeze(0), **INTR)
    frac = vis.float().mean().item()
    print(f"    {frac*100:.1f}% of cluster Gaussians flagged visible (rendered alpha max={alpha.max():.2f})")
    assert frac > 0.95, "Gaussians that render must be flagged visible"
    print("[ok] rasterizer-convention cross-check")


if __name__ == "__main__":
    test_in_front_vs_behind()
    test_fov_boundary()
    test_disagreement_signal_narrow_fov()
    test_direction_and_elevation_shift()
    test_diversity()
    test_matches_rasterizer_convention()
    print("\nAll visibility tests passed.")
