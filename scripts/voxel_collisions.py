"""
Check how many Gaussians collide in the same voxel cell at SplatFormer's grid_resolution.

Usage: python scripts/voxel_collisions.py <path/to/point_cloud.ply> [grid_res]
"""
import sys, numpy as np
from plyfile import PlyData

path = sys.argv[1]
G = int(sys.argv[2]) if len(sys.argv) > 2 else 384

ply = PlyData.read(path)
v = ply['vertex']
xyz = np.stack([v['x'], v['y'], v['z']], axis=1)   # already in [0,1] after MinMaxScaler
N = xyz.shape[0]

# Re-normalize defensively (the saved PLY may carry exact loader-normalized coords)
xyz_n = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0) + 1e-12)

# Voxel index
gc = np.floor(xyz_n * G).astype(np.int64)
gc = np.clip(gc, 0, G-1)
keys = gc[:, 0] * G * G + gc[:, 1] * G + gc[:, 2]     # flat cell index

unique_cells, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
occupied = unique_cells.size
total_cells = G ** 3

print(f"N Gaussians            : {N}")
print(f"grid_resolution G      : {G}    (G^3 = {total_cells:,} cells)")
print(f"unique occupied cells  : {occupied:,}")
print(f"occupancy ratio        : {occupied/total_cells*100:.4f}%")
print(f"avg Gaussians per cell : {N/occupied:.3f}")
print()
print(f"== distribution of Gaussians per occupied cell ==")
for bin_label, lo, hi in [
        ("=1", 1, 1), ("=2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10),
        ("11-50", 11, 50), ("51-100", 51, 100), (">100", 101, 10**9)]:
    n_cells = int(((counts >= lo) & (counts <= hi)).sum())
    n_gauss = int(counts[(counts >= lo) & (counts <= hi)].sum())
    print(f"  {bin_label:>7s}: {n_cells:>7d} cells   ({n_gauss:>7d} Gaussians)")

print()
colliding_gauss = int(N - (counts == 1).sum())
print(f"Gaussians sharing a cell with ≥1 other : {colliding_gauss}  ({colliding_gauss/N*100:.1f}%)")
print(f"max Gaussians in a single cell         : {counts.max()}")

# Show a few hot cells
top = np.argsort(counts)[-5:][::-1]
print()
print("Top 5 most-populated cells:")
for idx in top:
    cell_key = unique_cells[idx]
    ci, cj, ck = cell_key // (G*G), (cell_key // G) % G, cell_key % G
    print(f"  voxel ({ci:3d},{cj:3d},{ck:3d})  → {counts[idx]:4d} Gaussians")

# Also try a coarser grid to mimic the deeper pooling stages
print()
print(f"== Same analysis at G/2 = {G//2} (mimics encoder stage 1 effective resolution) ==")
gc2 = np.floor(xyz_n * (G//2)).astype(np.int64)
gc2 = np.clip(gc2, 0, G//2-1)
keys2 = gc2[:, 0] * (G//2)**2 + gc2[:, 1] * (G//2) + gc2[:, 2]
_, c2 = np.unique(keys2, return_counts=True)
print(f"  unique occupied        : {c2.size:,}")
print(f"  avg Gaussians per cell : {N/c2.size:.3f}")
print(f"  max per cell           : {c2.max()}")
