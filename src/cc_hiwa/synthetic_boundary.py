"""Known-truth synthetic data for testing hierarchical OT applicability.

The generator deliberately separates four ways an unpaired alignment problem
can depart from the GC-HiWA contract: additive noise, a mixture of two global
orthogonal maps (sample-level correspondence contamination), a changed target
group, and a non-orthogonal deformation.  Simply permuting target rows is not
included as a corruption: unpaired OT is permutation invariant, so that would
not test a real failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryScenario:
    """One frozen condition in the synthetic applicability benchmark."""

    name: str
    noise_std: float = 0.01
    mixed_transform_fraction: float = 0.0
    group_shift: float = 0.0
    nonorthogonal_scale: float = 0.0
    isospectral_nonorthogonal: bool = False
    cross_modal_target: bool = False
    determinant_sign: int = 1
    samples_per_group: int = 8

    def __post_init__(self) -> None:
        if self.determinant_sign not in {-1, 1}:
            raise ValueError("determinant_sign must be -1 or +1")
        if self.samples_per_group < 4:
            raise ValueError("samples_per_group must be at least four")
        if self.noise_std < 0 or self.group_shift < 0 or self.nonorthogonal_scale < 0:
            raise ValueError("noise and shift magnitudes must be non-negative")
        if not 0 <= self.mixed_transform_fraction <= 1:
            raise ValueError("mixed_transform_fraction must be in [0, 1]")
        if self.nonorthogonal_scale >= 0.8:
            raise ValueError("nonorthogonal_scale must be smaller than 0.8")
        if self.isospectral_nonorthogonal and self.nonorthogonal_scale > 0:
            raise ValueError(
                "isospectral_nonorthogonal and nonorthogonal_scale are alternative deformations"
            )


@dataclass(frozen=True)
class SyntheticPair:
    """Unpaired observations plus withheld truth used only for evaluation."""

    source: np.ndarray
    target: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray
    target_paired_truth: np.ndarray
    rotation_truth: np.ndarray
    target_permutation: np.ndarray
    mixed_transform_mask: np.ndarray
    has_single_rotation_truth: bool


@dataclass(frozen=True)
class VariableGroupScenario:
    """Known-truth valid pair whose number of geometric groups is not fixed."""

    name: str
    true_groups: int
    samples_per_group: int = 10
    noise_std: float = 0.05

    def __post_init__(self) -> None:
        if self.true_groups < 2:
            raise ValueError("true_groups must be at least two")
        if self.samples_per_group < 4:
            raise ValueError("samples_per_group must be at least four")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


def random_orthogonal(dimension: int, seed: int, determinant_sign: int = 1) -> np.ndarray:
    """Return a seeded orthogonal matrix in the requested determinant component."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if determinant_sign not in {-1, 1}:
        raise ValueError("determinant_sign must be -1 or +1")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    q = q @ np.diag(np.sign(np.diag(r)))
    current_sign = 1 if np.linalg.det(q) >= 0 else -1
    correction = np.eye(dimension)
    correction[-1, -1] = determinant_sign * current_sign
    return q @ correction


def _centers() -> np.ndarray:
    """An asymmetric, well-separated four-group geometry in three dimensions."""
    return np.asarray(
        [
            [-2.20, -1.20, -0.80],
            [-1.00, 2.40, 0.70],
            [2.10, -1.50, 1.20],
            [2.70, 1.90, -1.40],
        ],
        dtype=float,
    )


def make_synthetic_pair(scenario: BoundaryScenario, seed: int) -> SyntheticPair:
    """Generate a shuffled source/target pair and retain inaccessible truth.

    The target row order is always permuted.  The paired target and source
    labels are returned solely for post-fit scoring; fitting must use only the
    shuffled point clouds.
    """
    rng = np.random.default_rng(seed)
    centers = _centers()
    n_groups, dimension = centers.shape
    labels = np.repeat(np.arange(n_groups), scenario.samples_per_group)
    source = centers[labels] + rng.normal(0.0, 0.18, size=(labels.size, dimension))
    rotation = random_orthogonal(dimension, seed + 10_003, scenario.determinant_sign)
    target_paired = source @ rotation.T
    has_single_rotation_truth = not (
        scenario.cross_modal_target
        or scenario.nonorthogonal_scale > 0
        or scenario.isospectral_nonorthogonal
    )

    if scenario.cross_modal_target:
        # Deliberately retain the ambient dimension but remove the shared pose
        # geometry: this target is an independently generated modality, not a
        # shuffled or rotated version of source samples.
        target_centers = np.asarray(
            [
                [3.8, 0.3, -2.7],
                [-2.5, -3.7, 0.2],
                [0.4, 3.6, 3.1],
                [4.9, -1.8, 1.7],
            ],
            dtype=float,
        )
        target_paired = target_centers[labels] + rng.normal(0.0, 0.42, size=source.shape)

    if scenario.group_shift > 0:
        # One target group no longer has the source group's transformed centre.
        target_paired[labels == 0] += scenario.group_shift * np.asarray([1.0, -0.7, 0.5])

    if scenario.nonorthogonal_scale > 0:
        deformation = np.diag(
            [
                1.0 + scenario.nonorthogonal_scale,
                1.0 - scenario.nonorthogonal_scale,
                1.0 + 0.5 * scenario.nonorthogonal_scale,
            ]
        )
        target_paired = target_paired @ deformation.T

    if scenario.isospectral_nonorthogonal:
        # This construction is deliberately adversarial for a covariance-spectrum
        # check.  With S = X_c^T X_c and any orthogonal Q, A=S^(1/2) Q S^(-1/2)
        # obeys A S A^T = S, while A is generally not orthogonal whenever Q does
        # not commute with the anisotropic covariance.  Therefore the target has
        # exactly the same empirical covariance spectrum as the source but does
        # not admit a single orthogonal coordinate map.
        centered = source - source.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        square_root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        inverse_square_root = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        whitening_space_rotation = random_orthogonal(dimension, seed + 30_003, determinant_sign=1)
        deformation = square_root @ whitening_space_rotation @ inverse_square_root
        target_paired = source @ deformation.T

    mixed_mask = rng.random(labels.size) < scenario.mixed_transform_fraction
    if np.any(mixed_mask):
        alternative = random_orthogonal(dimension, seed + 20_003, scenario.determinant_sign)
        target_paired[mixed_mask] = source[mixed_mask] @ alternative.T

    target_paired = target_paired + rng.normal(0.0, scenario.noise_std, size=target_paired.shape)
    permutation = rng.permutation(labels.size)
    return SyntheticPair(
        source=source,
        target=target_paired[permutation],
        source_labels=labels,
        target_labels=labels[permutation],
        target_paired_truth=target_paired,
        rotation_truth=rotation,
        target_permutation=permutation,
        mixed_transform_mask=mixed_mask,
        has_single_rotation_truth=has_single_rotation_truth,
    )


def make_variable_group_pair(scenario: VariableGroupScenario, seed: int) -> SyntheticPair:
    """Create a shuffled valid orthogonal pair with an arbitrary true group count.

    Group labels and the paired target remain evaluation-only.  Centres are
    generated by a seeded rejection sampler so that the difficulty is not
    dominated by accidentally coincident components.
    """
    rng = np.random.default_rng(seed)
    dimension = 3
    centres: list[np.ndarray] = []
    while len(centres) < scenario.true_groups:
        candidate = rng.normal(size=dimension)
        candidate = 3.0 * candidate / np.linalg.norm(candidate)
        if all(np.linalg.norm(candidate - existing) >= 1.7 for existing in centres):
            centres.append(candidate)
    centre_matrix = np.asarray(centres, dtype=float)
    labels = np.repeat(np.arange(scenario.true_groups), scenario.samples_per_group)
    source = centre_matrix[labels] + rng.normal(0.0, 0.18, size=(labels.size, dimension))
    rotation = random_orthogonal(dimension, seed + 30_003, determinant_sign=1)
    target_paired = source @ rotation.T
    target_paired += rng.normal(0.0, scenario.noise_std, size=target_paired.shape)
    permutation = rng.permutation(labels.size)
    return SyntheticPair(
        source=source,
        target=target_paired[permutation],
        source_labels=labels,
        target_labels=labels[permutation],
        target_paired_truth=target_paired,
        rotation_truth=rotation,
        target_permutation=permutation,
        mixed_transform_mask=np.zeros(labels.size, dtype=bool),
        has_single_rotation_truth=True,
    )


def default_boundary_scenarios(samples_per_group: int = 8) -> tuple[BoundaryScenario, ...]:
    """Small, named screen spanning valid geometry and contract violations."""
    return (
        BoundaryScenario("ideal_rotation", samples_per_group=samples_per_group),
        BoundaryScenario("ideal_reflection", determinant_sign=-1, samples_per_group=samples_per_group),
        BoundaryScenario("noise_05", noise_std=0.05, samples_per_group=samples_per_group),
        BoundaryScenario("noise_10", noise_std=0.10, samples_per_group=samples_per_group),
        BoundaryScenario("moderate_noise", noise_std=0.20, samples_per_group=samples_per_group),
        BoundaryScenario("mixed_transforms_10pct", mixed_transform_fraction=0.10, samples_per_group=samples_per_group),
        BoundaryScenario("mixed_transforms_15pct", mixed_transform_fraction=0.15, samples_per_group=samples_per_group),
        BoundaryScenario("mixed_transforms_25pct", mixed_transform_fraction=0.25, samples_per_group=samples_per_group),
        BoundaryScenario("group_shift_030", group_shift=0.30, samples_per_group=samples_per_group),
        BoundaryScenario("group_shift_050", group_shift=0.50, samples_per_group=samples_per_group),
        BoundaryScenario("group_shift_1", group_shift=1.0, samples_per_group=samples_per_group),
        BoundaryScenario("nonorthogonal_05pct", nonorthogonal_scale=0.05, samples_per_group=samples_per_group),
        BoundaryScenario("nonorthogonal_10pct", nonorthogonal_scale=0.10, samples_per_group=samples_per_group),
        BoundaryScenario("nonorthogonal_30pct", nonorthogonal_scale=0.30, samples_per_group=samples_per_group),
        BoundaryScenario(
            "isospectral_nonorthogonal",
            noise_std=0.0,
            isospectral_nonorthogonal=True,
            samples_per_group=samples_per_group,
        ),
        BoundaryScenario("cross_modal_independent", cross_modal_target=True, samples_per_group=samples_per_group),
    )
