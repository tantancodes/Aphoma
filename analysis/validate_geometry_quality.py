"""Validate geometry-quality behavior with existing and synthetic meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.geometry_quality import (
    GeometryEvaluationSettings,
    TriangleMesh,
    evaluate_geometry,
    load_obj_mesh,
)


def write_obj(path: Path, mesh: TriangleMesh):
    with path.open("w", encoding="utf-8") as handle:
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for triangle in mesh.triangles:
            handle.write(f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n")


def synthetic_meshes(reference_path: Path, output: Path, seed: int):
    reference = load_obj_mesh(reference_path)
    rng = np.random.default_rng(seed)
    translated = TriangleMesh(reference.vertices + np.array([0.08, -0.05, 0.04]), reference.triangles)
    mild_noise = TriangleMesh(reference.vertices + rng.normal(0, 0.002, reference.vertices.shape), reference.triangles)
    strong_noise = TriangleMesh(reference.vertices + rng.normal(0, 0.01, reference.vertices.shape), reference.triangles)
    face_centers = reference.vertices[reference.triangles].mean(axis=1)
    retained = reference.triangles[face_centers[:, 0] <= np.quantile(face_centers[:, 0], 0.85)]
    cropped = TriangleMesh(reference.vertices, retained)
    meshes = {
        "rigid_translation": translated,
        "mild_vertex_noise": mild_noise,
        "strong_vertex_noise": strong_noise,
        "region_removal_15_percent_faces": cropped,
    }
    paths = {}
    for name, mesh in meshes.items():
        path = output / f"{name}.obj"
        write_obj(path, mesh)
        paths[name] = path
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--existing-candidate")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--sample-count", type=int, default=100_000)
    args = parser.parse_args()
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(args.reference).expanduser().resolve()
    settings = GeometryEvaluationSettings(evaluation_sample_count=args.sample_count)
    candidates = {"self": reference}
    if args.existing_candidate:
        candidates["existing_full_reference_repeat"] = Path(args.existing_candidate).expanduser().resolve()
    candidates.update(synthetic_meshes(reference, output, settings.random_seed))
    reports = {}
    for name, candidate in candidates.items():
        report = evaluate_geometry(reference, candidate, settings)
        reports[name] = report
        (output / f"{name}_geometry_quality.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        name: {
            "primary_y": report["symmetric_distances"]["primary_target_y_symmetric_rmse_normalized"],
            "symmetric_p95_normalized": report["symmetric_distances"]["p95_normalized"],
            "registration_fitness": report["registration"]["fitness"],
            "registration_rmse": report["registration"]["inlier_rmse"],
            "accuracy_at_0.5_percent_diagonal": report["threshold_metrics"]["0.005_reference_diagonal"]["accuracy_fraction"],
            "completeness_at_0.5_percent_diagonal": report["threshold_metrics"]["0.005_reference_diagonal"]["completeness_fraction"],
        }
        for name, report in reports.items()
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
