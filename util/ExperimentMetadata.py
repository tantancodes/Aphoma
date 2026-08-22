"""Experiment identity and perturbation metadata for reconstruction reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class ExperimentMetadata:
    artifact_id: str | None = None
    experiment_id: str | None = None
    parent_reference_run_id: str | None = None
    experiment_type: str | None = None
    perturbation_type: str | None = None
    perturbation_parameters: dict | None = None
    random_seed: int | None = None
    original_image_count: int | None = None
    images_used_count: int | None = None
    fraction_images_retained: float | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_perturbation_parameters(value) -> dict | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("perturbation_parameters must be a JSON object")
    return parsed


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_experiment_metadata(metadata, input_paths) -> ExperimentMetadata:
    """Resolve derived counts and manifest identity without changing any input files."""
    if metadata is None:
        metadata = ExperimentMetadata()
    elif isinstance(metadata, dict):
        metadata = ExperimentMetadata(**metadata)
    elif not isinstance(metadata, ExperimentMetadata):
        raise TypeError("experiment metadata must be ExperimentMetadata, dict, or None")

    images_used = sum(
        1 for path in input_paths
        if Path(path).is_file() and Path(path).suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff")
    )
    if metadata.images_used_count is not None and metadata.images_used_count != images_used:
        raise ValueError(
            f"images_used_count={metadata.images_used_count} does not match {images_used} reconstruction inputs"
        )
    original_count = metadata.original_image_count if metadata.original_image_count is not None else images_used
    if original_count < images_used:
        raise ValueError("original_image_count cannot be smaller than images_used_count")
    fraction = images_used / original_count if original_count else None

    manifest_path = None
    manifest_sha256 = None
    if metadata.manifest_path:
        path = Path(metadata.manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"experiment manifest not found: {path}")
        manifest_path = str(path)
        manifest_sha256 = _manifest_sha256(path)

    return replace(
        metadata,
        perturbation_parameters=parse_perturbation_parameters(metadata.perturbation_parameters),
        original_image_count=original_count,
        images_used_count=images_used,
        fraction_images_retained=fraction,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
