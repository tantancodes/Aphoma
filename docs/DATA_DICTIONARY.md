# Data dictionary

The candidate-level table is `analysis_outputs/fast_dense_geometry_dataset_v1/fast_dense_geometry_modeling_dataset_v1.csv`. Missing metrics remain explicit rather than fabricated.

## Identity and experiment fields

| Fields | Meaning |
|---|---|
| `artifact_id`, `experiment_id`, `run_id` | Physical object, condition, and unique execution. |
| `project_name`, `run_mode`, `execution_profile_version` | Reconstruction identity and profile. |
| `manifest_path`, `manifest_sha256` | Authoritative image list and content hash. |
| `random_seed`, `selection_type` | Reproducible selection identity. |
| `images_used_count`, `original_image_count`, `fraction_images_retained` | Input quantity. |
| `parent_reference_run_id` | Profile-matched reference relationship. |

## Raw-image and detector features

`image_*` fields summarize Laplacian sharpness, luminance, grayscale quantiles, contrast, entropy, edge density, and capped ORB counts using mean/median/SD/min/max suffixes. `orb_high_cap_*`, `fast_threshold_*`, and `sift_uncapped_*` summarize detector counts and density per analysis megapixel. Version/signature fields identify settings.

## Match-graph fields

`pair_good_matches_*` describes ORB ratio-test match distributions. Strong/weak fractions use 100/25-match thresholds. `graph_density`, degree fields, isolated nodes, component fields, normalized algebraic connectivity, and clustering describe graph topology. `adjacent_good_matches_*` describes support between consecutive retained acquisition-order images.

## Early-SfM fields

Camera alignment, tie-point, valid-tie-point, projection, and valid-observation fields are measured after error reduction. Normalized fields include tie points/image, valid observations/image, and observations/valid tie point. Reprojection fields report mean, median, RMSE, p95, and maximum pixels.

## Geometry targets

| Fields | Meaning |
|---|---|
| `geometry_y_normalized_symmetric_rmse` | Primary bidirectional surface RMSE/reference diagonal. |
| `candidate_to_reference_rmse_normalized` | Directional accuracy error. |
| `reference_to_candidate_rmse_normalized` | Directional completeness error. |
| `symmetric_*_normalized` | Median and p95/p99 bidirectional errors. |
| `accuracy_at_*`, `completeness_at_*` | Fractions within reference-diagonal thresholds. |

## Registration and QC

`registration_fitness` is the fraction within the ICP correspondence threshold. `registration_rmse` and its normalized form describe inlier residual. `registration_qc_flag` is pass/caution/manual-review and never an automatic exclusion. Evaluator version/signature fields preserve method identity.

