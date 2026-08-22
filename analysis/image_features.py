"""Interpretable raw-image features and manifest-level aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FEATURE_VERSION = 1
DEFAULT_MAX_DIMENSION = 1600
AGGREGATE_FEATURES = [
    "sharpness_laplacian_variance",
    "luminance_mean",
    "luminance_std",
    "grayscale_median",
    "grayscale_p05",
    "grayscale_p95",
    "contrast_p95_p05",
    "entropy_bits",
    "edge_density",
    "orb_keypoint_count",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis_image(image, max_dimension):
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def extract_image_features(path, max_dimension=DEFAULT_MAX_DIMENSION) -> dict:
    path = Path(path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    original_height, original_width = image.shape[:2]
    analysis_image, scale = _analysis_image(image, max_dimension)
    gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)
    p05, median, p95 = np.percentile(gray, [5, 50, 95])
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram[histogram > 0] / gray.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    edges = cv2.Canny(gray, 100, 200)
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints = orb.detect(gray, None)
    return {
        "feature_version": FEATURE_VERSION,
        "image_path": str(path),
        "filename": path.name,
        "sha256": sha256_file(path),
        "image_width_px": original_width,
        "image_height_px": original_height,
        "analysis_width_px": gray.shape[1],
        "analysis_height_px": gray.shape[0],
        "analysis_scale": scale,
        "sharpness_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "grayscale_median": float(median),
        "grayscale_p05": float(p05),
        "grayscale_p95": float(p95),
        "contrast_p95_p05": float(p95 - p05),
        "entropy_bits": entropy,
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "orb_keypoint_count": len(keypoints),
    }


def features_for_manifest(manifest_path, max_dimension=DEFAULT_MAX_DIMENSION) -> pd.DataFrame:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source_dataset"]).expanduser().resolve()
    rows = []
    for entry in manifest["images"]:
        row = extract_image_features(source / entry["filename"], max_dimension=max_dimension)
        if row["sha256"].lower() != entry["sha256"].lower():
            raise ValueError(f"manifest hash mismatch during feature extraction: {entry['filename']}")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_image_set(image_features: pd.DataFrame, manifest_path) -> dict:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [entry["filename"] for entry in manifest["images"]]
    subset = image_features.set_index("filename").loc[selected]
    row = {
        "artifact_id": manifest.get("artifact_id"),
        "experiment_id": manifest.get("experiment_id"),
        "manifest_path": str(manifest_path),
        "random_seed": manifest.get("random_seed"),
        "image_count": len(subset),
        "fraction_images_retained": manifest.get(
            "fraction_images_retained", len(subset) / manifest.get("original_image_count", len(subset))
        ),
    }
    for feature in AGGREGATE_FEATURES:
        values = subset[feature]
        for statistic, value in {
            "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1),
            "min": values.min(), "max": values.max(),
        }.items():
            row[f"image_{feature}_{statistic}"] = float(value)
    return row
