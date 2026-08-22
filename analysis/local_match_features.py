"""Versioned local-feature, pairwise-match, and image-graph measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from analysis.image_features import DEFAULT_MAX_DIMENSION, sha256_file


@dataclass(frozen=True)
class LocalFeatureConfig:
    version: int = 1
    max_dimension: int = DEFAULT_MAX_DIMENSION
    orb_high_cap: int = 20_000
    orb_fast_threshold: int = 20
    fast_threshold: int = 20
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    ratio_threshold: float = 0.75
    edge_min_good_matches: int = 100
    weak_pair_good_matches: int = 25

    def signature(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def load_gray(path, max_dimension):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    scale = min(1.0, max_dimension / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def compare_keypoint_detectors(path, config=LocalFeatureConfig()) -> dict:
    path = Path(path).expanduser().resolve()
    gray = load_gray(path, config.max_dimension)
    area_mpx = gray.size / 1_000_000
    detectors = {
        "orb_high_cap": cv2.ORB_create(
            nfeatures=config.orb_high_cap, fastThreshold=config.orb_fast_threshold
        ),
        "fast_threshold": cv2.FastFeatureDetector_create(
            threshold=config.fast_threshold, nonmaxSuppression=True
        ),
        "sift_uncapped": cv2.SIFT_create(
            nfeatures=0,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
        ),
    }
    row = {
        "local_feature_version": config.version,
        "local_feature_config_signature": config.signature(),
        "filename": path.name,
        "sha256": sha256_file(path),
        "analysis_width_px": gray.shape[1],
        "analysis_height_px": gray.shape[0],
        "analysis_area_megapixels": area_mpx,
    }
    for name, detector in detectors.items():
        keypoints = detector.detect(gray, None)
        row[f"{name}_keypoint_count"] = len(keypoints)
        row[f"{name}_keypoints_per_megapixel"] = len(keypoints) / area_mpx
    row["orb_high_cap_saturated"] = row["orb_high_cap_keypoint_count"] >= config.orb_high_cap
    return row


def orb_descriptors(path, cache_directory, config=LocalFeatureConfig()):
    path = Path(path).expanduser().resolve()
    cache_directory = Path(cache_directory).expanduser().resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    image_hash = sha256_file(path)
    cache_path = cache_directory / f"{image_hash}_{config.signature()}_orb.npz"
    if cache_path.is_file():
        cached = np.load(cache_path)
        return cached["keypoints"], cached["descriptors"]
    gray = load_gray(path, config.max_dimension)
    detector = cv2.ORB_create(
        nfeatures=config.orb_high_cap, fastThreshold=config.orb_fast_threshold
    )
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    coordinates = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    np.savez_compressed(cache_path, keypoints=coordinates, descriptors=descriptors)
    return coordinates, descriptors


def match_descriptor_pairs(image_paths, cache_directory, config=LocalFeatureConfig()) -> pd.DataFrame:
    paths = [Path(path).expanduser().resolve() for path in image_paths]
    descriptor_cache = {
        path.name: orb_descriptors(path, cache_directory, config)[1] for path in paths
    }
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    rows = []
    for first_index, first in enumerate(paths):
        first_descriptors = descriptor_cache[first.name]
        for second in paths[first_index + 1:]:
            second_descriptors = descriptor_cache[second.name]
            if len(first_descriptors) < 2 or len(second_descriptors) < 2:
                good_count = 0
            else:
                matches = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
                good_count = sum(
                    1 for match in matches
                    if len(match) == 2
                    and match[0].distance < config.ratio_threshold * match[1].distance
                )
            rows.append({
                "local_feature_version": config.version,
                "local_feature_config_signature": config.signature(),
                "filename_a": first.name,
                "filename_b": second.name,
                "good_match_count": good_count,
                "is_graph_edge": good_count >= config.edge_min_good_matches,
                "is_weak_pair": good_count < config.weak_pair_good_matches,
            })
    return pd.DataFrame(rows)


def _graph_statistics(nodes, edges):
    node_index = {node: index for index, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    neighbors = {node: set() for node in nodes}
    for first, second in edges:
        i, j = node_index[first], node_index[second]
        adjacency[i, j] = adjacency[j, i] = 1
        neighbors[first].add(second)
        neighbors[second].add(first)
    degrees = adjacency.sum(axis=1)
    components = []
    unseen = set(nodes)
    while unseen:
        start = next(iter(unseen))
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbors[node] - component)
        unseen -= component
        components.append(component)
    clustering = []
    for node in nodes:
        local = list(neighbors[node])
        if len(local) < 2:
            clustering.append(0.0)
            continue
        links = sum(1 for i, first in enumerate(local) for second in local[i + 1:] if second in neighbors[first])
        clustering.append(2 * links / (len(local) * (len(local) - 1)))
    laplacian = np.diag(degrees) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian) if len(nodes) > 1 else np.array([0.0])
    algebraic_connectivity = max(0.0, float(eigenvalues[1])) if len(eigenvalues) > 1 else 0.0
    possible_edges = len(nodes) * (len(nodes) - 1) / 2
    largest = max((len(component) for component in components), default=0)
    return {
        "graph_node_count": len(nodes),
        "graph_edge_count": len(edges),
        "graph_density": len(edges) / possible_edges if possible_edges else 0.0,
        "graph_mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "graph_min_degree": float(degrees.min()) if len(degrees) else 0.0,
        "graph_degree_std": float(degrees.std(ddof=0)) if len(degrees) else 0.0,
        "graph_min_degree_fraction": float(degrees.min() / (len(nodes) - 1)) if len(nodes) > 1 else 0.0,
        "graph_isolated_node_count": int(np.count_nonzero(degrees == 0)),
        "graph_connected_component_count": len(components),
        "graph_largest_component_size": largest,
        "graph_largest_component_fraction": largest / len(nodes) if nodes else 0.0,
        "graph_algebraic_connectivity": algebraic_connectivity,
        "graph_algebraic_connectivity_normalized": (
            algebraic_connectivity / len(nodes) if nodes else 0.0
        ),
        "graph_mean_clustering_coefficient": float(np.mean(clustering)) if clustering else 0.0,
    }


def aggregate_match_graph(pair_matches, manifest_path, canonical_order, config=LocalFeatureConfig()):
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = {entry["filename"] for entry in manifest["images"]}
    nodes = [filename for filename in canonical_order if filename in selected]
    pairs = pair_matches[
        pair_matches["filename_a"].isin(selected) & pair_matches["filename_b"].isin(selected)
    ]
    counts = pairs["good_match_count"].to_numpy(dtype=float)
    nonzero = counts[counts > 0]
    edges = list(
        pairs.loc[pairs["good_match_count"] >= config.edge_min_good_matches, ["filename_a", "filename_b"]]
        .itertuples(index=False, name=None)
    )
    row = {
        "artifact_id": manifest.get("artifact_id"),
        "experiment_id": manifest.get("experiment_id"),
        "local_feature_version": config.version,
        "local_feature_config_signature": config.signature(),
        "pair_evaluated_count": len(pairs),
        "pair_good_matches_mean": float(counts.mean()),
        "pair_good_matches_median": float(np.median(counts)),
        "pair_good_matches_p10": float(np.percentile(counts, 10)),
        "pair_good_matches_p90": float(np.percentile(counts, 90)),
        "pair_good_matches_min_nonzero": float(nonzero.min()) if len(nonzero) else 0.0,
        "pair_strong_fraction": float(np.mean(counts >= config.edge_min_good_matches)),
        "pair_weak_fraction": float(np.mean(counts < config.weak_pair_good_matches)),
    }
    row.update(_graph_statistics(nodes, edges))
    lookup = {
        frozenset((item.filename_a, item.filename_b)): item.good_match_count
        for item in pairs.itertuples(index=False)
    }
    adjacent_counts = np.array([
        lookup[frozenset((nodes[index], nodes[index + 1]))]
        for index in range(len(nodes) - 1)
    ], dtype=float)
    row.update({
        "acquisition_order_supported_by_exif": True,
        "adjacent_pair_count": len(adjacent_counts),
        "adjacent_good_matches_mean": float(adjacent_counts.mean()),
        "adjacent_good_matches_min": float(adjacent_counts.min()),
        "adjacent_weak_fraction": float(np.mean(adjacent_counts < config.weak_pair_good_matches)),
    })
    return row
