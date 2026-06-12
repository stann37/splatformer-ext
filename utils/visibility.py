"""
Per-Gaussian frustum-visibility features (T1.1).

For each Gaussian we compute, against the full train-view and test-view camera sets:
  - coverage: fraction of cameras whose frustum contains the Gaussian center
  - angular diversity: 1 - |mean viewing direction| over the cameras that see it
    (0 = all cameras see it from one direction, 1 = isotropic coverage)
and three derived signals:
  - disagreement = coverage_test - coverage_train
    (positive = rendered by OOD test views but weakly constrained by training views)
  - direction shift = 1 - cos(mean train viewing dir, mean test viewing dir), in [0,2]
    (catches azimuth-split OOD, e.g. MVImgNet front/side halves; degenerates to ~0
     for symmetric rings whose resultant collapses)
  - elevation shift = mean_z(test dirs) - mean_z(train dirs), in [-2,2]
    (catches elevation OOD, e.g. Objaverse-OOD low-ring train -> top-down test —
     the primary signal in object-centric scenes, where wide-FoV cameras contain
     the whole object in every frustum and coverage alone cannot discriminate)

Reconstruction-only: uses camera poses, never image content.

Camera convention matches utils/gs_utils.rasterize_gaussians_to_singleimg:
camera_to_worlds are OpenGL/Blender c2w; flip y,z columns to get OpenCV-style
projection with +z in front.
"""
import torch

VISIBILITY_DIM = 7


@torch.no_grad()
def frustum_visibility(
    means: torch.Tensor,            # (N, 3), same (normalized) space as the cameras
    camera_to_worlds: torch.Tensor, # (V, 3, 4) or (V, 4, 4), OpenGL convention
    fx, fy, cx, cy, width, height,
    near: float = 1e-4,
) -> torch.Tensor:                  # (N, V) bool
    """Frustum test (no occlusion)."""
    device = means.device
    c2w = camera_to_worlds.to(device).float()
    if c2w.shape[-2] == 3:
        bottom = torch.tensor([0, 0, 0, 1.0], device=device).expand(c2w.shape[0], 1, 4)
        c2w = torch.cat([c2w, bottom], dim=-2)                      # (V, 4, 4)

    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
    R = c2w[:, :3, :3] @ flip                                       # (V, 3, 3)
    T = c2w[:, :3, 3]                                               # (V, 3)
    # world -> camera: p_cam = R^T (p - T)
    pts = means.unsqueeze(0) - T.unsqueeze(1)                       # (V, N, 3)
    pts_cam = torch.einsum('vij,vnj->vni', R.transpose(1, 2), pts)  # (V, N, 3)

    z = pts_cam[..., 2]
    fx = float(fx); fy = float(fy); cx = float(cx); cy = float(cy)
    W = int(width); H = int(height)
    z_safe = torch.where(z.abs() < near, torch.full_like(z, near), z)
    u = fx * pts_cam[..., 0] / z_safe + cx
    v = fy * pts_cam[..., 1] / z_safe + cy
    visible = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)  # (V, N)
    return visible.transpose(0, 1)                                   # (N, V)


@torch.no_grad()
def _coverage_diversity_meandir(means, c2w, vis):
    """vis: (N, V) bool. Returns coverage (N,1), angular diversity (N,1), the
    normalized mean viewing direction (N,3) and the mean z-component of the viewing
    directions (N,1 — an elevation statistic that survives ring symmetry).
    Statistics are computed over the cameras that see the point; if none do, we fall
    back to all cameras (the *pose geometry* still tells us how this region would be
    looked at)."""
    device = means.device
    cam_pos = c2w[:, :3, 3].to(device).float()                       # (V, 3)
    coverage = vis.float().mean(dim=1, keepdim=True)                 # (N, 1)

    dirs = means.unsqueeze(1) - cam_pos.unsqueeze(0)                 # (N, V, 3)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
    n_vis = vis.float().sum(dim=1, keepdim=True)                     # (N, 1)
    use_all = (n_vis.squeeze(-1) == 0)                               # (N,)
    weights = vis.float()
    weights[use_all] = 1.0
    n_eff = weights.sum(dim=1, keepdim=True)
    resultant = (dirs * weights.unsqueeze(-1)).sum(dim=1)            # (N, 3)
    diversity = 1.0 - resultant.norm(dim=-1, keepdim=True) / (n_eff + 1e-8)
    mean_dir = resultant / (resultant.norm(dim=-1, keepdim=True) + 1e-8)
    mean_z = resultant[:, 2:3] / (n_eff + 1e-8)                      # (N, 1)
    return coverage, diversity, mean_dir, mean_z


@torch.no_grad()
def compute_visibility_features(
    means: torch.Tensor,                  # (N, 3) normalized space
    train_camera_to_worlds: torch.Tensor, # (V_tr, 3|4, 4) same space
    test_camera_to_worlds: torch.Tensor,  # (V_te, 3|4, 4)
    fx, fy, cx, cy, width, height,
) -> torch.Tensor:                        # (N, 7) float32
    """[coverage_train, coverage_test, disagreement,
        diversity_train, diversity_test, direction_shift, elevation_shift]"""
    def _as44(c):
        c = torch.as_tensor(c).float()
        if c.shape[-2] == 3:
            bottom = torch.tensor([0, 0, 0, 1.0]).expand(c.shape[0], 1, 4)
            c = torch.cat([c, bottom], dim=-2)
        return c

    c2w_tr = _as44(train_camera_to_worlds)
    c2w_te = _as44(test_camera_to_worlds)
    vis_tr = frustum_visibility(means, c2w_tr, fx, fy, cx, cy, width, height)
    vis_te = frustum_visibility(means, c2w_te, fx, fy, cx, cy, width, height)
    cov_tr, div_tr, dir_tr, z_tr = _coverage_diversity_meandir(means, c2w_tr, vis_tr)
    cov_te, div_te, dir_te, z_te = _coverage_diversity_meandir(means, c2w_te, vis_te)
    dir_shift = 1.0 - (dir_tr * dir_te).sum(dim=-1, keepdim=True)    # (N, 1) in [0, 2]
    elev_shift = z_te - z_tr                                          # (N, 1) in [-2, 2]
    return torch.cat(
        [cov_tr, cov_te, cov_te - cov_tr, div_tr, div_te, dir_shift, elev_shift], dim=1
    ).float()
