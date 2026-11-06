# Reproducibility

## Scope and environment

Manifest validation, feature extraction, geometry evaluation, and modeling can be repeated when source images and saved meshes/reports are available. Reconstructing also requires a compatible Agisoft Metashape Professional installation and valid license; this repository cannot provide or bypass proprietary software or licensing.

The validated macOS environment used Python 3.12 and Metashape API 2.2.1. Dependencies are listed in `requirements.txt` and `requirements-macos.txt`. Per-run reports record API/profile versions.

## Inputs and reconstruction metadata

`experiments/manifests/arf_dog_baseline_50.json` defines the canonical set; image-count and structured manifests are under `experiments/manifests/`. Each lists ordered filenames and SHA-256 hashes. Random selections record seed, method, and algorithm version. Resolve them with `util.ExperimentMetadata.resolve_manifest_inputs`; do not substitute files.

For licensed reconstruction, invoke `photogrammetryScripts.py` with `--manifest`, the intended `--run-mode`, and a fresh output directory. Reports record effective settings, experiment identity, stages, timing, and outputs. Do not treat geometry from different profiles as equivalent.

## Feature and report processing

- Raw-image features: `analysis/image_features.py`.
- Detector, ORB descriptor, pair matching, and graph logic: `analysis/local_match_features.py` and `analysis/run_local_match_analysis.py`.
- Report flattening and normalized early metrics: `analysis/report_dataset.py`.

Preserve canonical acquisition order, configuration signatures, and descriptor-cache keys. Predictor tables must exclude dense, target, registration, and QC fields.

## Geometry and modeling

Use `analysis/geometry_quality.py` with `GeometryEvaluationSettings()` and the fixed profile-matched reference. The validated signature is `d6c0d0c230cb34c3`. Preserve seed, sample counts, rigid-only ICP, and thresholds; save full JSON results including mesh hashes and registration QC.

The approved modeling table is Git-ignored under `analysis_outputs/`. Modeling uses small families, fold-local standardization, linear/Ridge estimators, leave-one-condition-out, trajectory-group, selection-family, and pass-QC sensitivity. Secondary targets do not select the primary model.

Before publication, verify row count, run IDs, evaluator signature, QC distribution, and every numerical claim directly against CSV/JSON outputs.
