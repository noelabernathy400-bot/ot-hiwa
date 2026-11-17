"""DAD author MATLAB port, cross-checked against nerdslab Python history.

Keeps 3D kNN-density KL grid search, translation search, and final reflected
2D angular search with the author's log-density discrepancy. Neither stage
uses class labels. The 3D display retains coarse z; decoding uses final xy.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .numerics import whiten, covariance_power


def direction_rotation(direction):
    target = direction / np.linalg.norm(direction)
    origin = np.array([0., 0., 1.])
    cosine = np.clip(origin @ target, -1, 1)
    axis = np.cross(origin, target)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1., -1., -1.])
    return Rotation.from_rotvec(axis / sine * np.arctan2(sine, cosine)).as_matrix()


def density(query, points, k):
    radii = cKDTree(points).query(query, k=[min(k, len(points))])[0][:, 0]
    values = np.maximum(radii, 1e-14) ** (-points.shape[1])
    return values / values.sum()


def grid_density_2d(x, bins=50):
    # MATLAB prob_grid.m uses a data-specific square domain and F flattening.
    a = np.linspace(x.min() - 1, x.max() + 1, bins)
    xx, yy = np.meshgrid(a, a)
    query = np.column_stack((xx.ravel(order="F"), yy.ravel(order="F")))
    return density(query, x, 3)


def dad(x, y, seed=0, angular_grid=5, density_bins=50, num_angles=90):
    rng = np.random.default_rng(seed)
    # Both input clouds are normalized by common benchmark preprocessing.
    axis = np.linspace(-1, 1, angular_grid)
    grid = np.meshgrid(axis, axis, axis)
    directions = np.column_stack([z.ravel(order="F") for z in grid])
    directions = np.vstack(([0, 0, 1], directions[np.linalg.norm(directions, axis=1) > .1]))
    axis = np.linspace(-4, 4, density_bins)
    grid = np.meshgrid(axis, axis, axis)
    query = np.column_stack([z.ravel(order="F") for z in grid])
    target_density = density(query, y, 5)
    losses = []
    best_loss, best_matrix = np.inf, None
    for direction in directions:
        matrix = direction_rotation(direction)
        rotated = x @ matrix  # MATLAB rotate_data uses row data times matrix.
        source_density = density(query, rotated, 5)
        loss = float(source_density @ np.log(source_density / target_density))
        losses.append(loss)
        if loss < best_loss:
            best_loss, best_matrix = loss, matrix
    coarse = x @ best_matrix
    # numT=1 coarse translation; author still performs final 1000 shifts.
    shifts = rng.standard_normal((1000, 3)) * .5
    tree = cKDTree(y)
    shift_losses = np.array([tree.query(coarse + t)[0].mean() for t in shifts])
    shift = shifts[np.argmin(shift_losses)]
    coarse += shift
    target_xy, target_mean, _ = whiten(y[:, :2])
    target_log = np.log(grid_density_2d(target_xy, density_bins))
    keep = np.ones((density_bins, density_bins), dtype=bool)
    mid = int(np.floor(density_bins / 2 + .5)) - 1
    keep[mid-1:mid+2, mid-1:mid+2] = False
    keep = keep.ravel(order="F")
    best = np.inf
    angular_losses = []
    for sign in [-1, 1]:
        for angle in np.linspace(0, np.pi, num_angles):
            c, s = np.cos(angle), np.sin(angle)
            matrix = np.diag([sign, 1]) @ np.array([[c, -s], [s, c]])
            rec, _, _ = whiten(coarse[:, :2] @ matrix)
            source_log = np.log(grid_density_2d(rec, density_bins))
            loss = np.linalg.norm((source_log-target_log)[keep]) / np.linalg.norm(target_log[keep])
            angular_losses.append(float(loss))
            if loss < best:
                best, final_xy = float(loss), rec
    # Convert final normalized 2D output back into common target coordinates.
    final_xy = final_xy @ covariance_power(np.cov(y[:, :2], rowvar=False), .5) + target_mean
    aligned = np.column_stack((final_xy, coarse[:, 2]))
    return {"aligned": aligned, "coarse_R": best_matrix.T,
            "translation": shift, "coarse_losses": np.array(losses),
            "angular_losses": np.array(angular_losses), "coarse_kl": best_loss,
            "final_log_density_error": best, "coarse_aligned": coarse}
