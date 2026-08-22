"""Read-only reconstruction metrics collection and report serialization."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import Metashape

from util.ExecutionProfiles import RunMode, get_execution_profile
from util.ExperimentMetadata import resolve_experiment_metadata


CSV_FIELDS = [
    "schema_version", "run_id", "project_name", "started_at_utc",
    "run_mode", "execution_profile_version",
    "artifact_id", "experiment_id", "parent_reference_run_id",
    "experiment_type", "perturbation_type", "perturbation_parameters",
    "random_seed", "original_image_count", "images_used_count",
    "fraction_images_retained", "manifest_path", "manifest_sha256", "notes",
    "finished_at_utc", "metashape_api_version", "input_image_count",
    "image_width_px", "image_height_px", "camera_model",
    "sparse_quality", "depth_model_quality", "depth_filter_mode",
    "effective_sparse_downscale", "effective_depth_downscale",
    "effective_depth_filter_mode", "effective_mesh_face_count_mode",
    "effective_mesh_face_count_custom", "effective_build_model",
    "effective_reorient_model", "effective_build_uv", "effective_build_texture",
    "effective_texture_size", "effective_texture_count", "effective_export_model",
    "effective_export_format", "planned_stages", "completed_stages", "skipped_stages",
    "stage_states",
    "mask_mode_value", "mask_mode_name", "palette", "total_cameras",
    "aligned_cameras", "unaligned_cameras", "alignment_fraction",
    "tie_point_count", "valid_tie_point_count", "projection_record_count",
    "valid_tie_point_observation_count", "reprojection_error_count",
    "reprojection_error_mean_px", "reprojection_error_median_px",
    "reprojection_error_rmse_px", "reprojection_error_p95_px",
    "reprojection_error_max_px", "depth_map_count", "model_vertex_count",
    "model_face_count", "texture_count", "texture_width_px",
    "texture_height_px", "model_succeeded", "texture_succeeded",
    "export_succeeded", "photo_matching_seconds", "camera_alignment_seconds",
    "photo_matching_alignment_seconds", "error_reduction_seconds",
    "depth_maps_seconds", "model_building_seconds", "uv_generation_seconds",
    "texture_building_seconds", "uv_texture_seconds", "export_seconds",
    "total_runtime_seconds", "metashape_project_path", "exported_model_path",
    "success", "error_code", "exception_type", "exception_message",
    "report_json_path",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _float_meta(meta: dict, key: str):
    try:
        return float(meta[key])
    except (KeyError, TypeError, ValueError):
        return None


def _meta_value(meta, key: str):
    try:
        return meta[key]
    except (KeyError, TypeError):
        return None


def _error_value(code):
    if code is None:
        return None
    return getattr(code, "value", code)


class ReconstructionMetrics:
    """Collects metrics without invoking or changing photogrammetry operations."""

    TASK_STAGES = {
        "MetashapeTask_AlignPhotos": "alignment",
        "MetashapeTask_ErrorReduction": "error_reduction",
        "MetashapeTask_DetectMarkers": "marker_detection",
        "MetashapeTask_AddScales": "scale_creation",
        "MetashapeTask_BuildModel": "dense_model",
        "MetashapeTask_Reorient": "reorientation",
        "MetashapeTask_BuildTextures": "uv_texture",
        "MetashapeTask_ExportModel": "export",
    }

    def __init__(self, project_name, input_paths, basedir, mask_mode, config, profile=None,
                 experiment_metadata=None):
        self.project_name = str(project_name)
        self.basedir = Path(basedir).resolve()
        self.run_id = str(uuid4())
        self.started_at = _utc_now()
        self.started_monotonic = time.perf_counter()
        self._stage_starts = {}
        self._stage_durations = {}
        self._failed = False
        self.profile = get_execution_profile(profile)

        valid_inputs = [
            Path(path).resolve() for path in input_paths
            if Path(path).is_file() and Path(path).suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff")
        ]
        self.experiment = resolve_experiment_metadata(experiment_metadata, valid_inputs)
        mask_value = getattr(mask_mode, "value", mask_mode)
        mask_name = getattr(mask_mode, "name", str(mask_mode) if mask_mode is not None else None)
        reports_dir = self.basedir / "reports"
        timestamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        self.report_path = reports_dir / f"{self.project_name}_{timestamp}_{self.run_id}.json"
        self.csv_path = self.basedir.parent / "reconstruction_summary.csv"

        self.report = {
            "schema_version": "1.2",
            "run_id": self.run_id,
            "experiment": self.experiment.to_dict(),
            "run": {
                "project_name": self.project_name,
                "run_mode": self.profile.run_mode.value,
                "execution_profile_version": self.profile.version,
                "effective_settings": self.profile.effective_settings(),
                "started_at_utc": _iso_utc(self.started_at),
                "finished_at_utc": None,
                "metashape_api_version": Metashape.app.version,
                "input_image_count": len(valid_inputs),
                "image_width_px": None,
                "image_height_px": None,
                "camera_model": None,
                "sparse_quality": self.profile.sparse_downscale,
                "depth_model_quality": self.profile.depth_downscale,
                "depth_filter_mode": self.profile.depth_filter_mode,
                "mask_mode": {"value": mask_value, "name": mask_name},
                "palette": config.getProperty("photogrammetry", "palette"),
            },
            "pipeline": {
                "planned_stages": self.profile.planned_stages,
                "completed_stages": [],
                "skipped_stages": self.profile.skipped_stages,
                "stage_states": {
                    stage: ("skipped" if stage in self.profile.skipped_stages else "not_reached")
                    for stage in self.TASK_STAGES.values()
                },
            },
            "alignment": {
                "snapshot_stage": None,
                "total_cameras": None,
                "aligned_cameras": None,
                "unaligned_cameras": None,
                "alignment_fraction": None,
                "tie_point_count": None,
                "valid_tie_point_count": None,
                "projection_record_count": None,
                "valid_tie_point_observation_count": None,
                "reprojection_error_px": {
                    "count": None, "mean": None, "median": None,
                    "rmse": None, "p95": None, "max": None,
                },
                "camera_error_statistics": None,
            },
            "dense_model": {
                "depth_map_count": None,
                "model_vertex_count": None,
                "model_face_count": None,
                "texture_count": None,
                "texture_width_px": None,
                "texture_height_px": None,
                "model_succeeded": None,
                "texture_succeeded": None,
                "export_succeeded": None,
            },
            "timing_seconds": {
                "photo_matching": None,
                "camera_alignment": None,
                "photo_matching_alignment": None,
                "error_reduction": None,
                "depth_maps": None,
                "model_building": None,
                "uv_generation": None,
                "texture_building": None,
                "uv_texture": None,
                "export": None,
                "total_runtime": None,
            },
            "output_status": {
                "metashape_project_path": str(self.basedir / f"{self.project_name}.psx"),
                "exported_model_path": None,
                "report_json_path": str(self.report_path),
                "summary_csv_path": str(self.csv_path),
                "success": False,
                "error": None,
            },
        }

    def stage_started(self, task):
        self._stage_starts[id(task)] = time.perf_counter()

    def stage_finished(self, task, success, code):
        start = self._stage_starts.pop(id(task), None)
        if start is not None:
            self._stage_durations[type(task).__name__] = time.perf_counter() - start
        if not bool(success):
            self._failed = True
            self.report["output_status"]["error"] = {
                "code": _error_value(code),
                "exception_type": None,
                "exception_message": None,
            }

        class_name = type(task).__name__
        stage = self.TASK_STAGES.get(class_name)
        if stage:
            state = "succeeded" if bool(success) else "failed"
            self.report["pipeline"]["stage_states"][stage] = state
            if bool(success) and stage not in self.report["pipeline"]["completed_stages"]:
                self.report["pipeline"]["completed_stages"].append(stage)
        if class_name == "MetashapeTask_AlignPhotos":
            self._collect_alignment(task.chunk, "post_alignment_pre_error_reduction")
        elif class_name == "MetashapeTask_ErrorReduction":
            self._collect_alignment(task.chunk, "post_error_reduction_pre_dense")
            self.report["timing_seconds"]["error_reduction"] = self._stage_durations.get(class_name)
        elif class_name == "MetashapeTask_BuildModel":
            self._collect_dense_model(task.chunk)
        elif class_name == "MetashapeTask_BuildTextures":
            self._collect_dense_model(task.chunk)
        elif class_name == "MetashapeTask_ExportModel":
            self.report["timing_seconds"]["export"] = self._stage_durations.get(class_name)
            output_path = Path(task.outputfolder, f"{task.outputfile}{task.extn}").resolve()
            self.report["output_status"]["exported_model_path"] = str(output_path)
            exported = bool(success) and output_path.is_file()
            self.report["dense_model"]["export_succeeded"] = exported
            if not exported:
                self._failed = True

    def stage_exception(self, task, exception):
        start = self._stage_starts.pop(id(task), None)
        if start is not None:
            self._stage_durations[type(task).__name__] = time.perf_counter() - start
        self._failed = True
        stage = self.TASK_STAGES.get(type(task).__name__)
        if stage:
            self.report["pipeline"]["stage_states"][stage] = "failed"
        self.report["output_status"]["error"] = {
            "code": None,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }

    def record_exception(self, exception):
        self._failed = True
        self.report["output_status"]["error"] = {
            "code": None,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        }

    def _collect_alignment(self, chunk, snapshot_stage):
        if chunk is None:
            return
        cameras = list(chunk.cameras)
        aligned = [camera for camera in cameras if camera.transform is not None]
        alignment = self.report["alignment"]
        alignment["snapshot_stage"] = snapshot_stage
        alignment["total_cameras"] = len(cameras)
        alignment["aligned_cameras"] = len(aligned)
        alignment["unaligned_cameras"] = len(cameras) - len(aligned)
        alignment["alignment_fraction"] = len(aligned) / len(cameras) if cameras else None

        dimensions = {(camera.sensor.width, camera.sensor.height) for camera in cameras if camera.sensor}
        if len(dimensions) == 1:
            width, height = next(iter(dimensions))
            self.report["run"]["image_width_px"] = width
            self.report["run"]["image_height_px"] = height
        models = {
            _meta_value(camera.sensor.meta, "Exif/Model") or _meta_value(camera.photo.meta, "Exif/Model")
            for camera in cameras if camera.sensor and camera.photo
        }
        models.discard(None)
        if len(models) == 1:
            self.report["run"]["camera_model"] = next(iter(models))

        tie_points = chunk.tie_points
        if tie_points is None:
            return
        points = list(tie_points.points)
        valid_points = {point.track_id: point for point in points if point.valid}
        alignment["tie_point_count"] = len(points)
        alignment["valid_tie_point_count"] = len(valid_points)

        projection_count = 0
        valid_observation_count = 0
        residuals = []
        for camera in cameras:
            for projection in tie_points.projections[camera]:
                projection_count += 1
                point = valid_points.get(projection.track_id)
                if point is None or camera.transform is None:
                    continue
                valid_observation_count += 1
                error = camera.error(point.coord, projection.coord)
                residuals.append(math.hypot(error.x, error.y))
        alignment["projection_record_count"] = projection_count
        alignment["valid_tie_point_observation_count"] = valid_observation_count
        if residuals:
            ordered = sorted(residuals)
            count = len(ordered)
            alignment["reprojection_error_px"] = {
                "count": count,
                "mean": sum(ordered) / count,
                "median": statistics.median(ordered),
                "rmse": math.sqrt(sum(value * value for value in ordered) / count),
                "p95": ordered[int(0.95 * (count - 1))],
                "max": ordered[-1],
            }

        match_seconds = _float_meta(tie_points.meta, "MatchPhotos/duration")
        align_seconds = _float_meta(chunk.meta, "AlignCameras/duration")
        timing = self.report["timing_seconds"]
        timing["photo_matching"] = match_seconds
        timing["camera_alignment"] = align_seconds
        if match_seconds is not None and align_seconds is not None:
            timing["photo_matching_alignment"] = match_seconds + align_seconds

    def _collect_dense_model(self, chunk):
        if chunk is None:
            return
        dense = self.report["dense_model"]
        try:
            dense["depth_map_count"] = sum(1 for _ in chunk.depth_maps.keys()) if chunk.depth_maps else 0
        except (AttributeError, TypeError):
            dense["depth_map_count"] = None

        model = chunk.model
        if model is None:
            return
        dense["model_vertex_count"] = len(model.vertices)
        dense["model_face_count"] = len(model.faces)
        dense["model_succeeded"] = bool(model.vertices and model.faces)
        timing = self.report["timing_seconds"]
        timing["depth_maps"] = _float_meta(model.meta, "BuildDepthMaps/duration")
        timing["model_building"] = _float_meta(model.meta, "BuildModel/duration")
        timing["uv_generation"] = _float_meta(model.meta, "BuildUV/duration")

        textures = list(model.textures)
        dense["texture_count"] = len(textures)
        if textures:
            image = textures[0].image()
            dense["texture_width_px"] = image.width
            dense["texture_height_px"] = image.height
            dense["texture_succeeded"] = image.width > 0 and image.height > 0
            timing["texture_building"] = _float_meta(textures[0].meta, "BuildTexture/duration")
        if timing["uv_generation"] is not None and timing["texture_building"] is not None:
            timing["uv_texture"] = timing["uv_generation"] + timing["texture_building"]

    def finalize(self):
        finished_at = _utc_now()
        self.report["run"]["finished_at_utc"] = _iso_utc(finished_at)
        self.report["timing_seconds"]["total_runtime"] = time.perf_counter() - self.started_monotonic
        stage_states = self.report["pipeline"]["stage_states"]
        required_stages = {
            RunMode.SFM_ONLY: ("alignment", "error_reduction"),
            RunMode.FAST_DENSE: ("alignment", "error_reduction", "dense_model"),
            RunMode.FULL_REFERENCE: ("alignment", "error_reduction", "dense_model", "uv_texture", "export"),
        }[self.profile.run_mode]
        success = not self._failed and all(stage_states[stage] == "succeeded" for stage in required_stages)
        self.report["output_status"]["success"] = success
        if success:
            self.report["output_status"]["error"] = None

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.report_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, self.report_path)
        self._append_csv()
        return self.report_path, self.csv_path

    def _append_csv(self):
        row = self._csv_row()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_csv_header_if_needed()
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def _migrate_csv_header_if_needed(self):
        """Extend an existing v1 CSV without discarding its previously collected rows."""
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == CSV_FIELDS:
                return
            rows = list(reader)
        temporary_path = self.csv_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for existing_row in rows:
                writer.writerow({field: existing_row.get(field) for field in CSV_FIELDS})
        os.replace(temporary_path, self.csv_path)

    def _csv_row(self):
        run = self.report["run"]
        alignment = self.report["alignment"]
        residual = alignment["reprojection_error_px"]
        dense = self.report["dense_model"]
        timing = self.report["timing_seconds"]
        status = self.report["output_status"]
        error = status["error"] or {}
        mask = run["mask_mode"]
        effective = run["effective_settings"]
        pipeline = self.report["pipeline"]
        experiment = self.report["experiment"]
        return {
            "schema_version": self.report["schema_version"], "run_id": self.run_id,
            "project_name": run["project_name"], "started_at_utc": run["started_at_utc"],
            "run_mode": run["run_mode"], "execution_profile_version": run["execution_profile_version"],
            "artifact_id": experiment["artifact_id"], "experiment_id": experiment["experiment_id"],
            "parent_reference_run_id": experiment["parent_reference_run_id"],
            "experiment_type": experiment["experiment_type"],
            "perturbation_type": experiment["perturbation_type"],
            "perturbation_parameters": json.dumps(experiment["perturbation_parameters"], sort_keys=True)
            if experiment["perturbation_parameters"] is not None else None,
            "random_seed": experiment["random_seed"],
            "original_image_count": experiment["original_image_count"],
            "images_used_count": experiment["images_used_count"],
            "fraction_images_retained": experiment["fraction_images_retained"],
            "manifest_path": experiment["manifest_path"],
            "manifest_sha256": experiment["manifest_sha256"], "notes": experiment["notes"],
            "finished_at_utc": run["finished_at_utc"], "metashape_api_version": run["metashape_api_version"],
            "input_image_count": run["input_image_count"], "image_width_px": run["image_width_px"],
            "image_height_px": run["image_height_px"], "camera_model": run["camera_model"],
            "sparse_quality": run["sparse_quality"], "depth_model_quality": run["depth_model_quality"],
            "depth_filter_mode": run["depth_filter_mode"], "mask_mode_value": mask["value"],
            "effective_sparse_downscale": effective["sparse_downscale"],
            "effective_depth_downscale": effective["depth_downscale"],
            "effective_depth_filter_mode": effective["depth_filter_mode"],
            "effective_mesh_face_count_mode": effective["mesh_face_count_mode"],
            "effective_mesh_face_count_custom": effective["mesh_face_count_custom"],
            "effective_build_model": effective["build_model"],
            "effective_reorient_model": effective["reorient_model"],
            "effective_build_uv": effective["build_uv"],
            "effective_build_texture": effective["build_texture"],
            "effective_texture_size": effective["texture_size"],
            "effective_texture_count": effective["texture_count"],
            "effective_export_model": effective["export_model"],
            "effective_export_format": effective["export_format"],
            "planned_stages": "|".join(pipeline["planned_stages"]),
            "completed_stages": "|".join(pipeline["completed_stages"]),
            "skipped_stages": "|".join(pipeline["skipped_stages"]),
            "stage_states": json.dumps(pipeline["stage_states"], sort_keys=True),
            "mask_mode_name": mask["name"], "palette": run["palette"],
            "total_cameras": alignment["total_cameras"], "aligned_cameras": alignment["aligned_cameras"],
            "unaligned_cameras": alignment["unaligned_cameras"], "alignment_fraction": alignment["alignment_fraction"],
            "tie_point_count": alignment["tie_point_count"], "valid_tie_point_count": alignment["valid_tie_point_count"],
            "projection_record_count": alignment["projection_record_count"],
            "valid_tie_point_observation_count": alignment["valid_tie_point_observation_count"],
            "reprojection_error_count": residual["count"], "reprojection_error_mean_px": residual["mean"],
            "reprojection_error_median_px": residual["median"], "reprojection_error_rmse_px": residual["rmse"],
            "reprojection_error_p95_px": residual["p95"], "reprojection_error_max_px": residual["max"],
            "depth_map_count": dense["depth_map_count"], "model_vertex_count": dense["model_vertex_count"],
            "model_face_count": dense["model_face_count"], "texture_count": dense["texture_count"],
            "texture_width_px": dense["texture_width_px"], "texture_height_px": dense["texture_height_px"],
            "model_succeeded": dense["model_succeeded"], "texture_succeeded": dense["texture_succeeded"],
            "export_succeeded": dense["export_succeeded"], "photo_matching_seconds": timing["photo_matching"],
            "camera_alignment_seconds": timing["camera_alignment"],
            "photo_matching_alignment_seconds": timing["photo_matching_alignment"],
            "error_reduction_seconds": timing["error_reduction"], "depth_maps_seconds": timing["depth_maps"],
            "model_building_seconds": timing["model_building"], "uv_generation_seconds": timing["uv_generation"],
            "texture_building_seconds": timing["texture_building"], "uv_texture_seconds": timing["uv_texture"],
            "export_seconds": timing["export"], "total_runtime_seconds": timing["total_runtime"],
            "metashape_project_path": status["metashape_project_path"],
            "exported_model_path": status["exported_model_path"], "success": status["success"],
            "error_code": error.get("code"), "exception_type": error.get("exception_type"),
            "exception_message": error.get("exception_message"), "report_json_path": status["report_json_path"],
        }
