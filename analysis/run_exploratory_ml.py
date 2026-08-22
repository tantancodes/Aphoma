"""Run the exploratory reconstruction proxy-target modeling workflow."""

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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.image_features import AGGREGATE_FEATURES, aggregate_image_set, features_for_manifest
from analysis.report_dataset import load_reconstruction_reports


PROXY_TARGETS = [
    "observations_per_valid_tie_point",
    "valid_observations_per_image",
    "reprojection_rmse_px",
    "tie_points_per_image",
]
QUALITY_PREDICTORS = [
    "image_sharpness_laplacian_variance_mean",
    "image_luminance_mean_mean",
    "image_luminance_std_mean",
    "image_entropy_bits_mean",
    "image_edge_density_mean",
    "image_orb_keypoint_count_mean",
]


def _models(seed):
    scaled_linear = lambda estimator: Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear": scaled_linear(LinearRegression()),
        "ridge": scaled_linear(Ridge(alpha=1.0)),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=500, max_depth=3, random_state=seed, n_jobs=-1
            )),
        ]),
    }


def build_model_results(modeling, seed):
    feature_sets = {
        "image_count_only": ["image_count"],
        "image_count_plus_quality": ["image_count", *QUALITY_PREDICTORS],
    }
    loo = LeaveOneOut()
    metrics, predictions, coefficients, importances = [], [], [], []
    for target in PROXY_TARGETS:
        y = modeling[target].to_numpy(dtype=float)
        for feature_set, features in feature_sets.items():
            X = modeling[features]
            for model_name, model in _models(seed).items():
                predicted = cross_val_predict(model, X, y, cv=loo, n_jobs=None)
                metrics.append({
                    "target": target,
                    "feature_set": feature_set,
                    "model": model_name,
                    "n": len(y),
                    "validation": "leave_one_condition_out",
                    "mae": mean_absolute_error(y, predicted),
                    "rmse": mean_squared_error(y, predicted) ** 0.5,
                    "r2": np.nan,
                    "r2_note": (
                        f"omitted: n={len(y)} is too small for a statistically meaningful "
                        "generalization claim"
                    ),
                })
                for index, actual, estimate in zip(modeling.index, y, predicted):
                    predictions.append({
                        "run_id": modeling.loc[index, "run_id"],
                        "experiment_id": modeling.loc[index, "experiment_id"],
                        "image_count": modeling.loc[index, "image_count"],
                        "target": target,
                        "feature_set": feature_set,
                        "model": model_name,
                        "actual": actual,
                        "predicted": estimate,
                        "residual_actual_minus_predicted": actual - estimate,
                    })
                model.fit(X, y)
                if model_name in {"linear", "ridge"}:
                    estimator = model.named_steps["model"]
                    for feature, coefficient in zip(features, estimator.coef_):
                        coefficients.append({
                            "target": target, "feature_set": feature_set,
                            "model": model_name, "feature": feature,
                            "standardized_x_coefficient": coefficient,
                            "note": f"descriptive full-data fit; unstable with n={len(y)}",
                        })
                if model_name == "random_forest":
                    estimator = model.named_steps["model"]
                    for feature, importance in zip(features, estimator.feature_importances_):
                        importances.append({
                            "target": target, "feature_set": feature_set,
                            "feature": feature, "importance": importance,
                            "note": f"descriptive full-data fit; highly unstable with n={len(y)}",
                        })
    return tuple(pd.DataFrame(rows) for rows in (metrics, predictions, coefficients, importances))


def save_eda(modeling, image_features, output):
    sns.set_theme(style="whitegrid")
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    image_features[AGGREGATE_FEATURES].hist(figsize=(15, 12), bins=12)
    plt.suptitle("Canonical image-feature distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(plot_dir / "image_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close()

    correlation_columns = ["image_count", *QUALITY_PREDICTORS, *PROXY_TARGETS]
    correlations = modeling[correlation_columns].corr(numeric_only=True)
    plt.figure(figsize=(13, 10))
    sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1)
    plt.title(f"Image-set features and proxy-target correlations (n={len(modeling)})")
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_matrix.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for target, axis in zip(PROXY_TARGETS, axes.ravel()):
        sns.lineplot(data=modeling, x="image_count", y=target, marker="o", ax=axis)
        axis.set_title(f"Image count vs proxy: {target}")
    plt.tight_layout()
    plt.savefig(plot_dir / "image_count_vs_proxy_targets.png", dpi=160)
    plt.close()

    selected = [
        "image_sharpness_laplacian_variance_mean",
        "image_entropy_bits_mean",
        "image_edge_density_mean",
        "image_orb_keypoint_count_mean",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for feature, axis in zip(selected, axes.ravel()):
        sns.regplot(data=modeling, x=feature, y="observations_per_valid_tie_point", ax=axis, ci=None)
        axis.set_title(f"{feature} vs observation multiplicity proxy")
    plt.tight_layout()
    plt.savefig(plot_dir / "selected_image_features_vs_observation_proxy.png", dpi=160)
    plt.close()
    return correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--manifest-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = load_reconstruction_reports(args.report_root)
    successful = reports[reports["success"] == True].copy()  # noqa: E712
    failed = reports[reports["success"] != True].copy()  # noqa: E712
    reports.to_csv(output / "reconstruction_runs_all.csv", index=False)
    successful.to_csv(output / "reconstruction_runs_successful.csv", index=False)
    failed.to_csv(output / "reconstruction_runs_failed.csv", index=False)

    image_features = features_for_manifest(args.baseline_manifest)
    image_features.to_csv(output / "image_features.csv", index=False)
    manifest_paths = sorted(Path(args.manifest_directory).glob("*.json"))
    set_features = pd.DataFrame(aggregate_image_set(image_features, path) for path in manifest_paths)
    set_features.to_csv(output / "image_set_features.csv", index=False)

    experimental = successful[
        (successful["experiment_type"] == "image_count_perturbation")
        & successful["manifest_path"].notna()
    ].copy()
    set_features_for_join = set_features.drop(
        columns=["manifest_path", "random_seed", "fraction_images_retained"]
    )
    modeling = experimental.merge(
        set_features_for_join, on=["artifact_id", "experiment_id"],
        how="inner", validate="one_to_one"
    )
    modeling = modeling.sort_values("image_count", ascending=False).reset_index(drop=True)
    if modeling.empty or modeling["experiment_id"].nunique() != len(modeling):
        raise ValueError("expected one successful reconstruction per experimental condition")
    modeling.to_csv(output / "modeling_table_proxy_targets.csv", index=False)
    modeling.describe(include="all").transpose().to_csv(output / "modeling_descriptive_statistics.csv")
    pd.DataFrame({
        "column": modeling.columns,
        "missing_count": modeling.isna().sum().values,
        "missing_fraction": modeling.isna().mean().values,
        "unique_non_null": modeling.nunique(dropna=True).values,
    }).to_csv(output / "modeling_missingness.csv", index=False)

    correlations = save_eda(modeling, image_features, output)
    correlations.to_csv(output / "correlation_matrix.csv")
    metrics, predictions, coefficients, importances = build_model_results(modeling, args.seed)
    metrics.to_csv(output / "model_comparison_loo.csv", index=False)
    predictions.to_csv(output / "cross_validated_predictions.csv", index=False)
    coefficients.to_csv(output / "linear_ridge_coefficients.csv", index=False)
    importances.to_csv(output / "random_forest_importances.csv", index=False)

    leakage_audit = {
        "proxy_targets": PROXY_TARGETS,
        "predictor_source": "raw images and manifest image count only",
        "excluded_as_leakage": [
            "tie_point_count", "valid_tie_point_count", "valid_observation_count",
            "projection_record_count", "all reprojection statistics", "all dense outcomes",
            "all reconstruction timings",
        ],
        "deterministically_redundant": ["fraction_images_retained (image_count / 50)"],
        "validation": (
            f"leave-one-experiment-condition-out; n={len(modeling)}; "
            f"artifacts={modeling['artifact_id'].nunique()}"
        ),
        "target_warning": "Targets are early-SfM proxy outcomes, not final reconstruction-quality ground truth.",
    }
    (output / "leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2) + "\n")
    print(json.dumps({
        "all_reports": len(reports), "successful_reports": len(successful),
        "failed_reports": len(failed), "image_rows": len(image_features),
        "experimental_modeling_rows": len(modeling), "output_directory": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
