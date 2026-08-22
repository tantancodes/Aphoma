"""Extract local-feature and match-graph features, then model proxy outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.image_features import aggregate_image_set, features_for_manifest
from analysis.local_match_features import (
    LocalFeatureConfig,
    aggregate_match_graph,
    compare_keypoint_detectors,
    match_descriptor_pairs,
)
from analysis.report_dataset import load_reconstruction_reports


PROXY_TARGETS = [
    "observations_per_valid_tie_point",
    "valid_observations_per_image",
    "reprojection_rmse_px",
    "tie_points_per_image",
]
GLOBAL_PREDICTORS = [
    "image_sharpness_laplacian_variance_mean",
    "image_luminance_std_mean",
    "image_edge_density_mean",
]
GRAPH_PREDICTORS = [
    "pair_good_matches_median",
    "graph_density",
    "adjacent_good_matches_mean",
]


def _models():
    def scaled(model):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ])
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": scaled(LinearRegression()),
        "ridge": scaled(Ridge(alpha=1.0)),
    }


def _validate_acquisition_order(source, entries):
    timestamps = []
    for entry in entries:
        with Image.open(source / entry["filename"]) as image:
            exif = image.getexif()
            timestamp = exif.get(36867) or exif.get(306)
        if timestamp is None:
            raise ValueError(f"missing EXIF acquisition time: {entry['filename']}")
        timestamps.append(timestamp)
    if timestamps != sorted(timestamps):
        raise ValueError("canonical manifest order is not nondecreasing by EXIF acquisition time")


def aggregate_detector_set(detector_comparison, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selected = [entry["filename"] for entry in manifest["images"]]
    subset = detector_comparison.set_index("filename").loc[selected]
    row = {
        "artifact_id": manifest.get("artifact_id"),
        "experiment_id": manifest.get("experiment_id"),
    }
    for detector in ("orb_high_cap", "fast_threshold", "sift_uncapped"):
        for measure in ("keypoint_count", "keypoints_per_megapixel"):
            values = subset[f"{detector}_{measure}"]
            row[f"{detector}_{measure}_mean"] = float(values.mean())
            row[f"{detector}_{measure}_min"] = float(values.min())
    return row


def build_model_comparison(modeling):
    feature_sets = {
        "A_image_count_only": ["image_count"],
        "B_global_quality_only": GLOBAL_PREDICTORS,
        "C_match_graph_only": GRAPH_PREDICTORS,
        "D_image_count_plus_graph": ["image_count", *GRAPH_PREDICTORS],
        "E_count_global_graph": ["image_count", *GLOBAL_PREDICTORS, *GRAPH_PREDICTORS],
    }
    loo = LeaveOneOut()
    metrics, predictions, coefficients = [], [], []
    for target in PROXY_TARGETS:
        y = modeling[target].to_numpy(dtype=float)
        for family, predictors in feature_sets.items():
            X = modeling[predictors]
            for model_name, model in _models().items():
                predicted = cross_val_predict(model, X, y, cv=loo)
                metrics.append({
                    "target": target, "predictor_family": family, "model": model_name,
                    "predictor_count": len(predictors), "n": len(y),
                    "rank_deficient_design": len(predictors) >= len(y) - 1,
                    "mae": mean_absolute_error(y, predicted),
                    "rmse": mean_squared_error(y, predicted) ** 0.5,
                    "r2": np.nan,
                    "validation": "leave_one_experiment_condition_out",
                })
                for index, actual, estimate in zip(modeling.index, y, predicted):
                    predictions.append({
                        "run_id": modeling.loc[index, "run_id"],
                        "experiment_id": modeling.loc[index, "experiment_id"],
                        "image_count": modeling.loc[index, "image_count"],
                        "target": target, "predictor_family": family, "model": model_name,
                        "actual": actual, "predicted": estimate,
                        "residual_actual_minus_predicted": actual - estimate,
                    })
                if model_name in {"linear", "ridge"}:
                    model.fit(X, y)
                    for predictor, coefficient in zip(predictors, model.named_steps["model"].coef_):
                        coefficients.append({
                            "target": target, "predictor_family": family,
                            "model": model_name, "predictor": predictor,
                            "standardized_x_coefficient": coefficient,
                            "note": f"descriptive full-data fit; n={len(y)}",
                        })
    return tuple(pd.DataFrame(rows) for rows in (metrics, predictions, coefficients))


def save_plots(modeling, detector_comparison, model_metrics, output):
    sns.set_theme(style="whitegrid")
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    detector_columns = [
        "orb_high_cap_keypoints_per_megapixel",
        "fast_threshold_keypoints_per_megapixel",
        "sift_uncapped_keypoints_per_megapixel",
    ]
    detector_comparison[detector_columns].plot.box(figsize=(11, 6), rot=15)
    plt.ylabel("Keypoints per analysis megapixel")
    plt.title("Local-detector density comparison across canonical images")
    plt.tight_layout()
    plt.savefig(plots / "detector_density_comparison.png", dpi=160)
    plt.close()

    graph_columns = [
        "pair_good_matches_mean", "pair_good_matches_median", "pair_strong_fraction",
        "graph_density", "graph_min_degree_fraction", "graph_algebraic_connectivity_normalized",
        "adjacent_good_matches_mean", "adjacent_good_matches_min",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    for column, axis in zip(graph_columns, axes.ravel()):
        sns.lineplot(data=modeling, x="image_count", y=column, marker="o", ax=axis)
        axis.set_title(column)
    plt.tight_layout()
    plt.savefig(plots / "graph_features_vs_image_count.png", dpi=160)
    plt.close()

    ridge = model_metrics[model_metrics["model"] == "ridge"].pivot(
        index="target", columns="predictor_family", values="rmse"
    )
    normalized = ridge.div(ridge["A_image_count_only"], axis=0)
    plt.figure(figsize=(11, 5))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="vlag", center=1)
    plt.title("Ridge LOO RMSE relative to image-count-only (lower is better)")
    plt.tight_layout()
    plt.savefig(plots / "ridge_relative_rmse.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--manifest-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = LocalFeatureConfig()

    baseline_path = Path(args.baseline_manifest).expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source = Path(baseline["source_dataset"]).expanduser().resolve()
    entries = baseline["images"]
    _validate_acquisition_order(source, entries)
    image_paths = [source / entry["filename"] for entry in entries]
    canonical_order = [entry["filename"] for entry in entries]

    detector_comparison = pd.DataFrame(
        compare_keypoint_detectors(path, config) for path in image_paths
    )
    detector_comparison.to_csv(output / "detector_comparison.csv", index=False)
    pair_matches = match_descriptor_pairs(image_paths, output / "descriptor_cache", config)
    pair_matches.to_csv(output / "canonical_pair_matches.csv", index=False)

    manifest_paths = sorted(Path(args.manifest_directory).glob("*.json"))
    detector_sets = pd.DataFrame(
        aggregate_detector_set(detector_comparison, path) for path in manifest_paths
    )
    detector_sets.to_csv(output / "detector_set_features.csv", index=False)
    graph_features = pd.DataFrame(
        aggregate_match_graph(pair_matches, path, canonical_order, config)
        for path in manifest_paths
    )
    graph_features.to_csv(output / "match_graph_features.csv", index=False)
    basic_images = features_for_manifest(baseline_path)
    basic_sets = pd.DataFrame(aggregate_image_set(basic_images, path) for path in manifest_paths)

    reports = load_reconstruction_reports(args.report_root)
    experimental = reports[
        (reports["success"] == True)  # noqa: E712
        & (reports["experiment_type"] == "image_count_perturbation")
    ].copy()
    join_columns = ["artifact_id", "experiment_id"]
    basic_join = basic_sets.drop(columns=["manifest_path", "random_seed", "fraction_images_retained"])
    modeling = experimental.merge(basic_join, on=join_columns, validate="one_to_one")
    modeling = modeling.merge(detector_sets, on=join_columns, validate="one_to_one")
    modeling = modeling.merge(graph_features, on=join_columns, validate="one_to_one")
    modeling = modeling.sort_values("image_count", ascending=False).reset_index(drop=True)
    if modeling.empty or modeling["experiment_id"].nunique() != len(modeling):
        raise ValueError("expected one successful run per image-count condition")
    modeling.to_csv(output / "modeling_table_with_graph_features.csv", index=False)

    metrics, predictions, coefficients = build_model_comparison(modeling)
    metrics.to_csv(output / "graph_model_comparison_loo.csv", index=False)
    predictions.to_csv(output / "graph_model_predictions_loo.csv", index=False)
    coefficients.to_csv(output / "graph_model_coefficients.csv", index=False)
    save_plots(modeling, detector_comparison, metrics, output)
    (output / "local_feature_settings.json").write_text(
        json.dumps({**config.__dict__, "config_signature": config.signature()}, indent=2) + "\n"
    )
    print(json.dumps({
        "detector_image_count": len(detector_comparison),
        "evaluated_pair_count": len(pair_matches),
        "experimental_condition_count": len(modeling),
        "output_directory": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
