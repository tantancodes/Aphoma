"""Load immutable reconstruction reports into tidy tabular data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _get(data, path, default=np.nan):
    for key in path.split("."):
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data if data is not None else default


REPORT_FIELDS = {
    "schema_version": "schema_version",
    "run_id": "run_id",
    "project_name": "run.project_name",
    "run_mode": "run.run_mode",
    "execution_profile_version": "run.execution_profile_version",
    "started_at_utc": "run.started_at_utc",
    "finished_at_utc": "run.finished_at_utc",
    "metashape_api_version": "run.metashape_api_version",
    "artifact_id": "experiment.artifact_id",
    "experiment_id": "experiment.experiment_id",
    "parent_reference_run_id": "experiment.parent_reference_run_id",
    "experiment_type": "experiment.experiment_type",
    "perturbation_type": "experiment.perturbation_type",
    "random_seed": "experiment.random_seed",
    "original_image_count": "experiment.original_image_count",
    "images_used_count": "experiment.images_used_count",
    "fraction_images_retained": "experiment.fraction_images_retained",
    "manifest_path": "experiment.manifest_path",
    "manifest_sha256": "experiment.manifest_sha256",
    "notes": "experiment.notes",
    "input_image_count": "run.input_image_count",
    "image_width_px": "run.image_width_px",
    "image_height_px": "run.image_height_px",
    "camera_model": "run.camera_model",
    "sparse_quality": "run.sparse_quality",
    "depth_model_quality": "run.depth_model_quality",
    "depth_filter_mode": "run.depth_filter_mode",
    "mask_mode": "run.mask_mode.name",
    "palette": "run.palette",
    "total_cameras": "alignment.total_cameras",
    "aligned_cameras": "alignment.aligned_cameras",
    "unaligned_cameras": "alignment.unaligned_cameras",
    "alignment_fraction": "alignment.alignment_fraction",
    "tie_point_count": "alignment.tie_point_count",
    "valid_tie_point_count": "alignment.valid_tie_point_count",
    "projection_record_count": "alignment.projection_record_count",
    "valid_observation_count": "alignment.valid_tie_point_observation_count",
    "reprojection_mean_px": "alignment.reprojection_error_px.mean",
    "reprojection_median_px": "alignment.reprojection_error_px.median",
    "reprojection_rmse_px": "alignment.reprojection_error_px.rmse",
    "reprojection_p95_px": "alignment.reprojection_error_px.p95",
    "reprojection_max_px": "alignment.reprojection_error_px.max",
    "depth_map_count": "dense_model.depth_map_count",
    "model_vertex_count": "dense_model.model_vertex_count",
    "model_face_count": "dense_model.model_face_count",
    "texture_count": "dense_model.texture_count",
    "model_succeeded": "dense_model.model_succeeded",
    "texture_succeeded": "dense_model.texture_succeeded",
    "export_succeeded": "dense_model.export_succeeded",
    "matching_seconds": "timing_seconds.photo_matching",
    "alignment_seconds": "timing_seconds.camera_alignment",
    "matching_alignment_seconds": "timing_seconds.photo_matching_alignment",
    "error_reduction_seconds": "timing_seconds.error_reduction",
    "depth_map_seconds": "timing_seconds.depth_maps",
    "model_seconds": "timing_seconds.model_building",
    "uv_texture_seconds": "timing_seconds.uv_texture",
    "export_seconds": "timing_seconds.export",
    "total_runtime_seconds": "timing_seconds.total_runtime",
    "success": "output_status.success",
    "error_code": "output_status.error.code",
    "error_type": "output_status.error.exception_type",
    "error_message": "output_status.error.exception_message",
    "project_path": "output_status.metashape_project_path",
    "exported_model_path": "output_status.exported_model_path",
}


def report_to_row(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    row = {name: _get(report, source) for name, source in REPORT_FIELDS.items()}
    row["report_path"] = str(path.resolve())
    row["perturbation_parameters"] = json.dumps(
        _get(report, "experiment.perturbation_parameters", None), sort_keys=True
    )
    if pd.isna(row["run_mode"]):
        row["run_mode"] = "FULL_REFERENCE" if bool(row["export_succeeded"]) else "UNKNOWN"
    if pd.isna(row["images_used_count"]):
        row["images_used_count"] = row["input_image_count"]
    if pd.isna(row["original_image_count"]):
        row["original_image_count"] = row["images_used_count"]
    if pd.isna(row["fraction_images_retained"]):
        original = row["original_image_count"]
        row["fraction_images_retained"] = row["images_used_count"] / original if original else np.nan

    images = row["images_used_count"]
    valid_ties = row["valid_tie_point_count"]
    observations = row["valid_observation_count"]
    row["tie_points_per_image"] = row["tie_point_count"] / images if images else np.nan
    row["valid_observations_per_image"] = observations / images if images else np.nan
    row["observations_per_valid_tie_point"] = observations / valid_ties if valid_ties else np.nan
    row["run_status"] = "successful_observation" if bool(row["success"]) else "failed_preflight_or_run"

    experiment_id = row["experiment_id"]
    if pd.isna(experiment_id):
        if row["images_used_count"] == 50 and row["sparse_quality"] == 1:
            experiment_id = "legacy-baseline-50-sparse1"
        else:
            experiment_id = f"legacy-unclassified:{row['project_name']}"
    row["condition_id"] = experiment_id
    return row


def load_reconstruction_reports(report_root) -> pd.DataFrame:
    paths = sorted(Path(report_root).glob("**/reports/*.json"))
    frame = pd.DataFrame(report_to_row(path) for path in paths)
    if frame.empty:
        return frame
    frame["condition_repeat_index"] = frame.groupby("condition_id", dropna=False).cumcount() + 1
    numeric = [
        column for column in frame.columns
        if column.endswith(("_count", "_fraction", "_px", "_seconds", "_per_image", "_per_valid_tie_point"))
        or column in {"images_used_count", "original_image_count", "random_seed", "sparse_quality", "depth_model_quality"}
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["started_at_utc", "run_id"], na_position="last").reset_index(drop=True)
