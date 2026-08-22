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


def resolve_manifest_inputs(manifest_path, input_directory=None) -> list[Path]:
    """Return the manifest's ordered, hash-validated source image paths."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"experiment manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("experiment manifest must contain a non-empty images list")

    declared_count = manifest.get("images_used_count", manifest.get("image_count"))
    if declared_count is not None and declared_count != len(images):
        raise ValueError(
            f"manifest image count {declared_count} does not match {len(images)} image entries"
        )

    source_value = manifest.get("source_dataset") or input_directory
    if not source_value:
        raise ValueError("manifest must define source_dataset or be used with an input directory")
    source_directory = Path(source_value).expanduser().resolve()
    if not source_directory.is_dir():
        raise FileNotFoundError(f"manifest source dataset not found: {source_directory}")

    resolved = []
    seen = set()
    for index, item in enumerate(images):
        if not isinstance(item, dict):
            raise ValueError(f"manifest image entry {index} must be an object")
        filename = item.get("filename")
        expected_hash = item.get("sha256")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"manifest image entry {index} has no filename")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"manifest image entry {filename} has an invalid SHA-256")
        if filename in seen:
            raise ValueError(f"duplicate manifest image filename: {filename}")
        seen.add(filename)

        image_path = (source_directory / filename).resolve()
        try:
            image_path.relative_to(source_directory)
        except ValueError as exception:
            raise ValueError(f"manifest image escapes source dataset: {filename}") from exception
        if not image_path.is_file():
            raise FileNotFoundError(f"manifest image not found: {image_path}")
        actual_hash = _manifest_sha256(image_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                f"SHA-256 mismatch for {image_path}: expected {expected_hash}, got {actual_hash}"
            )
        resolved.append(image_path)
    return resolved


def resolve_experiment_metadata(metadata, input_paths) -> ExperimentMetadata:
    """Resolve derived counts and manifest identity without changing any input files."""
    if metadata is None:
        metadata = ExperimentMetadata()
    elif isinstance(metadata, dict):
        metadata = ExperimentMetadata(**metadata)
    elif not isinstance(metadata, ExperimentMetadata):
        raise TypeError("experiment metadata must be ExperimentMetadata, dict, or None")

    manifest_path = None
    manifest_sha256 = None
    if metadata.manifest_path:
        path = Path(metadata.manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"experiment manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_values = {
            "artifact_id": manifest.get("artifact_id"),
            "experiment_id": manifest.get("experiment_id"),
            "experiment_type": manifest.get("experiment_type"),
            "perturbation_type": manifest.get("perturbation_type"),
            "perturbation_parameters": manifest.get("perturbation_parameters"),
            "random_seed": manifest.get("random_seed"),
            "original_image_count": manifest.get("original_image_count"),
            "images_used_count": manifest.get("images_used_count", manifest.get("image_count")),
        }
        updates = {}
        for field, manifest_value in manifest_values.items():
            supplied_value = getattr(metadata, field)
            if supplied_value is not None and manifest_value is not None and supplied_value != manifest_value:
                raise ValueError(
                    f"{field}={supplied_value!r} conflicts with manifest value {manifest_value!r}"
                )
            if supplied_value is None and manifest_value is not None:
                updates[field] = manifest_value
        metadata = replace(metadata, **updates)
        manifest_path = str(path)
        manifest_sha256 = _manifest_sha256(path)

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

    return replace(
        metadata,
        perturbation_parameters=parse_perturbation_parameters(metadata.perturbation_parameters),
        original_image_count=original_count,
        images_used_count=images_used,
        fraction_images_retained=fraction,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
