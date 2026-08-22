"""Reference-relative 3D surface quality evaluation.

The evaluator deliberately performs rigid registration only.  It never rescales a
candidate, because scale disagreement is part of reconstruction error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryEvaluationSettings:
    evaluator_version: int = 1
    schema_version: str = "1.0"
    random_seed: int = 20260822
    evaluation_sample_count: int = 100_000
    registration_sample_count: int = 20_000
    registration_method: str = "centroid_initialization_then_rigid_point_to_point_icp"
    icp_max_iterations: int = 60
    icp_relative_tolerance: float = 1e-7
    icp_absolute_tolerance_fraction: float = 1e-12
    icp_max_correspondence_distance_fraction: float = 0.05
    threshold_fractions_of_reference_diagonal: tuple[float, ...] = (
        0.0025, 0.005, 0.01, 0.02
    )

    def signature(self) -> str:
        payload = repr(sorted(asdict(self).items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_obj_mesh(path) -> TriangleMesh:
    """Load OBJ geometry only; materials, normals, and textures are ignored."""
    path = Path(path).expanduser().resolve()
    vertices, triangles = [], []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append(tuple(float(value) for value in values[1:4]))
            elif line.startswith("f "):
                indices = [int(item.split("/")[0]) for item in line.split()[1:]]
                resolved = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                for offset in range(1, len(resolved) - 1):
                    triangles.append((resolved[0], resolved[offset], resolved[offset + 1]))
    mesh = TriangleMesh(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"OBJ has no usable triangle mesh: {path}")
    if mesh.triangles.min() < 0 or mesh.triangles.max() >= len(mesh.vertices):
        raise ValueError(f"OBJ contains an invalid face index: {path}")
    return mesh


def sample_surface(mesh: TriangleMesh, count: int, seed: int) -> np.ndarray:
    """Sample triangle surfaces in proportion to area, deterministically."""
    triangle_vertices = mesh.vertices[mesh.triangles]
    cross = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    usable = areas > np.finfo(float).eps
    if not np.any(usable):
        raise ValueError("mesh has no nondegenerate triangles")
    triangle_vertices, areas = triangle_vertices[usable], areas[usable]
    rng = np.random.default_rng(seed)
    choices = rng.choice(len(areas), size=count, p=areas / areas.sum())
    chosen = triangle_vertices[choices]
    first = np.sqrt(rng.random(count))
    second = rng.random(count)
    return (
        (1 - first)[:, None] * chosen[:, 0]
        + (first * (1 - second))[:, None] * chosen[:, 1]
        + (first * second)[:, None] * chosen[:, 2]
    )


def _rigid_transform(source: np.ndarray, target: np.ndarray):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    covariance = np.einsum(
        "ni,nj->ij", source - source_center, target - target_center,
        optimize=False,
    )
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_center - np.einsum("ij,j->i", rotation, source_center)
    return rotation, translation


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ji->nj", points, transform[:3, :3]) + transform[:3, 3]


def rigid_icp(
    source, target, max_distance, max_iterations, relative_tolerance,
    absolute_tolerance=0.0,
):
    """Centroid-initialized, correspondence-trimmed point-to-point rigid ICP."""
    transform = np.eye(4)
    transform[:3, 3] = target.mean(axis=0) - source.mean(axis=0)
    target_tree = cKDTree(target)
    previous_rmse = np.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        transformed = apply_transform(source, transform)
        distances, indices = target_tree.query(transformed, workers=1)
        accepted = distances <= max_distance
        if accepted.sum() < 3:
            break
        rotation, translation = _rigid_transform(transformed[accepted], target[indices[accepted]])
        update = np.eye(4)
        update[:3, :3], update[:3, 3] = rotation, translation
        transform = update @ transform
        rmse = float(np.sqrt(np.mean(np.square(distances[accepted]))))
        if rmse <= absolute_tolerance or (
            np.isfinite(previous_rmse)
            and abs(previous_rmse - rmse) <= relative_tolerance * max(previous_rmse, 1e-15)
        ):
            break
        previous_rmse = rmse
    transformed = apply_transform(source, transform)
    distances, _ = target_tree.query(transformed, workers=1)
    accepted = distances <= max_distance
    return {
        "transform": transform,
        "fitness": float(accepted.mean()),
        "inlier_rmse": float(np.sqrt(np.mean(np.square(distances[accepted])))) if accepted.any() else None,
        "iterations": iterations,
        "correspondence_count": int(accepted.sum()),
    }


def _distance_summary(distances: np.ndarray, reference_diagonal: float) -> dict:
    return {
        "mean": float(distances.mean()),
        "median": float(np.median(distances)),
        "rmse": float(np.sqrt(np.mean(np.square(distances)))),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "max": float(distances.max()),
        "mean_normalized": float(distances.mean() / reference_diagonal),
        "rmse_normalized": float(np.sqrt(np.mean(np.square(distances))) / reference_diagonal),
        "p95_normalized": float(np.percentile(distances, 95) / reference_diagonal),
        "p99_normalized": float(np.percentile(distances, 99) / reference_diagonal),
    }


def evaluate_geometry(reference_mesh, candidate_mesh, settings=GeometryEvaluationSettings()):
    """Evaluate candidate geometry relative to a fixed reference mesh.

    Distances are Monte Carlo approximations to nearest-surface distance using
    deterministic, area-weighted surface samples. Candidate-to-reference is
    accuracy; reference-to-candidate is completeness.
    """
    reference_path = Path(reference_mesh).expanduser().resolve()
    candidate_path = Path(candidate_mesh).expanduser().resolve()
    reference = load_obj_mesh(reference_path)
    candidate = load_obj_mesh(candidate_path)
    reference_diagonal = float(np.linalg.norm(np.ptp(reference.vertices, axis=0)))
    if reference_diagonal <= 0:
        raise ValueError("reference bounding-box diagonal must be positive")

    registration_reference = sample_surface(reference, settings.registration_sample_count, settings.random_seed)
    registration_candidate = sample_surface(candidate, settings.registration_sample_count, settings.random_seed)
    raw_distances, _ = cKDTree(registration_reference).query(registration_candidate, workers=-1)
    max_distance = settings.icp_max_correspondence_distance_fraction * reference_diagonal
    registration = rigid_icp(
        registration_candidate, registration_reference, max_distance,
        settings.icp_max_iterations, settings.icp_relative_tolerance,
        settings.icp_absolute_tolerance_fraction * reference_diagonal,
    )

    reference_points = sample_surface(reference, settings.evaluation_sample_count, settings.random_seed + 1)
    candidate_points = sample_surface(candidate, settings.evaluation_sample_count, settings.random_seed + 1)
    candidate_points = apply_transform(candidate_points, registration["transform"])
    candidate_to_reference, _ = cKDTree(reference_points).query(candidate_points, workers=1)
    reference_to_candidate, _ = cKDTree(candidate_points).query(reference_points, workers=1)
    symmetric = np.concatenate((candidate_to_reference, reference_to_candidate))

    thresholds = {}
    for fraction in settings.threshold_fractions_of_reference_diagonal:
        distance = fraction * reference_diagonal
        thresholds[f"{fraction:.6g}_reference_diagonal"] = {
            "distance": distance,
            "accuracy_fraction": float(np.mean(candidate_to_reference <= distance)),
            "completeness_fraction": float(np.mean(reference_to_candidate <= distance)),
        }
    warnings = [
        "Reference-relative quality only; the reference mesh is not external ground truth.",
        "Nearest-surface distances are deterministic Monte Carlo approximations from sampled surfaces.",
        "No scale correction was applied; scale disagreement remains part of measured error.",
    ]
    if registration["fitness"] < 0.8:
        warnings.append("Registration fitness is below 0.8; inspect alignment before using quality metrics.")
    return {
        "schema_version": settings.schema_version,
        "evaluator_version": settings.evaluator_version,
        "evaluator_settings_signature": settings.signature(),
        "settings": asdict(settings),
        "reference": {
            "identifier": reference_path.stem,
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            "vertex_count": len(reference.vertices),
            "triangle_count": len(reference.triangles),
            "bounding_box_diagonal": reference_diagonal,
        },
        "candidate": {
            "identifier": candidate_path.stem,
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "vertex_count": len(candidate.vertices),
            "triangle_count": len(candidate.triangles),
        },
        "sampled_point_counts": {
            "registration_each_mesh": settings.registration_sample_count,
            "evaluation_each_mesh": settings.evaluation_sample_count,
        },
        "registration": {
            "method": settings.registration_method,
            "scale_allowed": False,
            "max_correspondence_distance": max_distance,
            "raw_candidate_to_reference_rmse": float(np.sqrt(np.mean(np.square(raw_distances)))),
            "fitness": registration["fitness"],
            "inlier_rmse": registration["inlier_rmse"],
            "iterations": registration["iterations"],
            "correspondence_count": registration["correspondence_count"],
            "transform_candidate_to_reference": registration["transform"].tolist(),
            "rotation_determinant": float(np.linalg.det(registration["transform"][:3, :3])),
        },
        "directional_distances": {
            "candidate_to_reference_accuracy": _distance_summary(candidate_to_reference, reference_diagonal),
            "reference_to_candidate_completeness": _distance_summary(reference_to_candidate, reference_diagonal),
        },
        "symmetric_distances": {
            **_distance_summary(symmetric, reference_diagonal),
            "chamfer_mean_squared": float((np.mean(candidate_to_reference ** 2) + np.mean(reference_to_candidate ** 2)) / 2),
            "chamfer_rmse": float(np.sqrt((np.mean(candidate_to_reference ** 2) + np.mean(reference_to_candidate ** 2)) / 2)),
            "primary_target_y_symmetric_rmse_normalized": float(
                np.sqrt((np.mean(candidate_to_reference ** 2) + np.mean(reference_to_candidate ** 2)) / 2)
                / reference_diagonal
            ),
        },
        "threshold_metrics": thresholds,
        "warnings": warnings,
    }
