# Methods

## Instrumentation and execution profiles

`util/ReconstructionMetrics.py` writes one JSON report per run and appends one flattened CSV row. Schema 1.2 includes UUID `run_id`, experiment metadata, effective settings, planned/completed/skipped stages, stage states (`succeeded`, `failed`, `skipped`, `not_reached`), early-SfM and model measurements, timings, paths, and structured failures. Missing values remain null. Early measurements are captured after error reduction and before dense reconstruction.

All version-1 profiles use the shared pipeline and sparse downscale 1.

| Profile | Verified effective stages/settings |
|---|---|
| `SFM_ONLY` | Alignment and error reduction; no marker/scale, dense model, reorientation, UV, texture, or export. |
| `FAST_DENSE` | Alignment, error reduction, marker/scale, depth downscale 4, Mild filtering, High mesh mode, safe reorientation; no UV, texture, or pipeline export. |
| `FULL_REFERENCE` | Alignment, error reduction, marker/scale, depth downscale 2, Mild filtering, High mesh mode, reorientation, one 4096×4096 texture, OBJ export. |

`FULL_REFERENCE` is the backward-compatible default.

## Experiment identity and manifests

An artifact identifies the physical object; an experiment identifies a condition; a run identifies one execution. Reports store artifact/experiment/run identity, parent reference, perturbation metadata, seed, counts, retained fraction, and manifest identity.

When supplied, a manifest is authoritative. Before Metashape starts, `resolve_manifest_inputs()` checks the nonempty list, declared count, filename uniqueness, containment inside the source directory, file existence, and every SHA-256. Directory input remains available without a manifest.

## Repeatability

Early-SfM repeatability used nine observations with identical 50-image sparse settings: three `FULL_REFERENCE`, one `FAST_DENSE`, and five `SFM_ONLY` runs. Profile-specific dense geometry was not pooled. Three full-reference runs estimated dense/runtime variability. A deterministic 50-vs-50 evaluator self-check produced normalized symmetric RMSE `5.16×10⁻17`.

## Image and local features

`analysis/image_features.py` resizes only for analysis to maximum dimension 1600. It records SHA-256 and computes grayscale Laplacian-variance sharpness, luminance mean/standard deviation, median/p05/p95, p95–p05 contrast, entropy, Canny edge density, and capped ORB count (5,000). Candidate summaries use distribution statistics across selected images.

The versioned local configuration uses maximum dimension 1600, ORB cap 20,000 and FAST threshold 20, standalone FAST threshold 20, uncapped SIFT with contrast threshold 0.04 and edge threshold 10, Lowe ratio 0.75, graph-edge threshold 100 good matches, and weak-pair threshold 25. Pair matching uses ORB descriptors and brute-force Hamming k-nearest neighbors (`k=2`). Cached descriptors are keyed by image SHA and configuration.

Images form graph nodes; pairs with at least 100 good matches form undirected edges. Features cover match distributions, strong/weak fractions, density, degree, components, normalized algebraic connectivity, clustering, and adjacent acquisition-order support. Canonical acquisition order is checked against EXIF time.

## FAST_DENSE experiment design

The fixed reference is the 50-image FAST_DENSE reconstruction. Candidates include nested random subsets at 40, 35, 30, and 25 images for seeds 20260822, 20260823, and 20260824, plus four structured 30-image selections. All candidates share the FAST_DENSE profile, avoiding profile-confounded target comparisons.

## Geometry evaluator

Evaluator version 1, schema 1.0, signature `d6c0d0c230cb34c3` uses deterministic seed 20260822, 20,000 area-weighted registration samples per mesh, 100,000 evaluation samples per mesh, centroid initialization and rigid point-to-point ICP, at most 60 iterations, relative tolerance `1×10⁻7`, absolute tolerance `1×10⁻12` of reference diagonal, maximum correspondence distance 5% of that diagonal, and thresholds at 0.25%, 0.5%, 1%, and 2%.

Registration permits rotation and translation only—never candidate-specific scale or non-rigid deformation. The primary target is bidirectional surface RMSE divided by the reference bounding-box diagonal. Candidate→reference measures reference-relative accuracy; reference→candidate measures completeness. Median, p95, p99, thresholds, and directional RMSEs are secondary. The reference is a reconstruction, not external ground truth.

QC is `pass` at fitness ≥0.95, `caution` at 0.80–0.95, and `manual_review` below 0.80. Low-fitness observations remain in the primary analysis.

## Modeling and leakage prevention

The n=16 analysis compares deliberately small dummy, image-count, reprojection, early-SfM, graph, count-plus-graph, and SfM-plus-graph families. Constant alignment and obvious redundant predictors were excluded. Linear and Ridge models standardize continuous predictors inside each training fold; Ridge alpha is 1.0. No dense/model, geometry-target, registration, or QC field enters a predictor set.

Validation includes leave-one-condition-out (interpolation-oriented), trajectory-group holdout, selection-family holdout (structural extrapolation), and pass-QC sensitivity. MAE, RMSE, and Spearman association are reported. Secondary targets are separate from primary selection. R² and training fit are not emphasized.

