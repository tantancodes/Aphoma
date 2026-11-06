# Experiments

## A. Curated 50-image baseline

One unsuitable frame (`_MG_4916.jpg`) was excluded from the original 51 JPGs. The 50-image baseline used no palette, no masks, sparse downscale 1, depth downscale 2, Mild filtering, High mesh behavior, one 4096×4096 texture, and OBJ export. It established that the macOS environment, licensed Metashape API, reconstruction stages, and export worked end to end.

## B. Repeatability

Three `FULL_REFERENCE` executions estimate full-pipeline variability. Identical early-SfM conditions were observed nine times by including the early stages of one FAST_DENSE and five SFM_ONLY executions. This separates sparse numerical variation from runtime variation and profile-specific geometry.

## C. First image-count SfM sweep

Nested manifest subsets retained 50, 45, 40, 35, 30, and 25 images using seed 20260822 and one deterministic shuffle. SFM_ONLY isolated early measurements without dense computation. Normalized observations prevented lower raw counts from being interpreted automatically as lower quality.

## D. Random FAST_DENSE subsets

Counts 40, 35, 30, and 25 were reconstructed across seeds 20260822, 20260823, and 20260824, nested within seed. Three independent trajectories expose between-subset variability at fixed count and prevent one arbitrary subset from masquerading as an image-count effect.

## E. Structured 30-image conditions

| Condition | Controlled purpose |
|---|---|
| Central contiguous gap | Creates localized missing acquisition coverage while count remains 30. |
| Uniform thinning | Distributes retained frames across acquisition order. |
| Remove weakest overlap | Removes low-support images using only pre-dense ORB graph node scores. |
| Remove strongest overlap | Contrast condition testing whether global overlap strength equals unique coverage. |

All are evaluated against the same 50-image FAST_DENSE reference with the same evaluator. FULL_REFERENCE and FAST_DENSE geometry are never pooled. The final modeling table contains 12 random and four structured candidates, each with unique `run_id` and condition-specific `experiment_id`, all for artifact `arf-dog`.

