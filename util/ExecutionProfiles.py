"""Named, versioned execution profiles for the shared reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


PROFILE_VERSION = 1

PIPELINE_STAGES = (
    "alignment",
    "error_reduction",
    "marker_detection",
    "scale_creation",
    "dense_model",
    "reorientation",
    "uv_texture",
    "export",
)


class RunMode(str, Enum):
    SFM_ONLY = "SFM_ONLY"
    FAST_DENSE = "FAST_DENSE"
    FULL_REFERENCE = "FULL_REFERENCE"


@dataclass(frozen=True)
class ExecutionProfile:
    run_mode: RunMode
    sparse_downscale: int
    depth_downscale: int | None
    depth_filter_mode: str | None
    mesh_face_count_mode: str | None
    mesh_face_count_custom: int | None
    build_model: bool
    reorient_model: bool
    build_uv: bool
    build_texture: bool
    texture_size: int | None
    texture_count: int | None
    export_model: bool
    export_format: str | None
    marker_and_scale: bool

    @property
    def version(self) -> int:
        return PROFILE_VERSION

    @property
    def planned_stages(self) -> list[str]:
        stages = ["alignment", "error_reduction"]
        if self.marker_and_scale:
            stages.extend(("marker_detection", "scale_creation"))
        if self.build_model:
            stages.append("dense_model")
        if self.reorient_model:
            stages.append("reorientation")
        if self.build_uv or self.build_texture:
            stages.append("uv_texture")
        if self.export_model:
            stages.append("export")
        return stages

    @property
    def skipped_stages(self) -> list[str]:
        planned = set(self.planned_stages)
        return [stage for stage in PIPELINE_STAGES if stage not in planned]

    def effective_settings(self) -> dict:
        return {
            "sparse_downscale": self.sparse_downscale,
            "depth_downscale": self.depth_downscale,
            "depth_filter_mode": self.depth_filter_mode,
            "mesh_face_count_mode": self.mesh_face_count_mode,
            "mesh_face_count_custom": self.mesh_face_count_custom,
            "build_model": self.build_model,
            "reorient_model": self.reorient_model,
            "build_uv": self.build_uv,
            "build_texture": self.build_texture,
            "texture_size": self.texture_size,
            "texture_count": self.texture_count,
            "export_model": self.export_model,
            "export_format": self.export_format,
        }


_PROFILES = {
    RunMode.SFM_ONLY: ExecutionProfile(
        run_mode=RunMode.SFM_ONLY,
        sparse_downscale=1,
        depth_downscale=None,
        depth_filter_mode=None,
        mesh_face_count_mode=None,
        mesh_face_count_custom=None,
        build_model=False,
        reorient_model=False,
        build_uv=False,
        build_texture=False,
        texture_size=None,
        texture_count=None,
        export_model=False,
        export_format=None,
        marker_and_scale=False,
    ),
    RunMode.FAST_DENSE: ExecutionProfile(
        run_mode=RunMode.FAST_DENSE,
        sparse_downscale=1,
        depth_downscale=4,
        depth_filter_mode="Mild",
        mesh_face_count_mode="High",
        mesh_face_count_custom=None,
        build_model=True,
        reorient_model=True,
        build_uv=False,
        build_texture=False,
        texture_size=None,
        texture_count=None,
        export_model=False,
        export_format=None,
        marker_and_scale=True,
    ),
    RunMode.FULL_REFERENCE: ExecutionProfile(
        run_mode=RunMode.FULL_REFERENCE,
        sparse_downscale=1,
        depth_downscale=2,
        depth_filter_mode="Mild",
        mesh_face_count_mode="High",
        mesh_face_count_custom=None,
        build_model=True,
        reorient_model=True,
        build_uv=True,
        build_texture=True,
        texture_size=4096,
        texture_count=1,
        export_model=True,
        export_format=".obj",
        marker_and_scale=True,
    ),
}


def get_execution_profile(run_mode=RunMode.FULL_REFERENCE) -> ExecutionProfile:
    """Resolve a profile name or enum, defaulting callers to FULL_REFERENCE."""
    if isinstance(run_mode, ExecutionProfile):
        return run_mode
    if run_mode is None:
        run_mode = RunMode.FULL_REFERENCE
    if isinstance(run_mode, str):
        run_mode = RunMode(run_mode.upper())
    return _PROFILES[run_mode]
