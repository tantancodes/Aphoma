# Limitations

- **One physical artifact:** object material, shape, capture, and background are confounded with artifact identity; cross-artifact generalization is untested.
- **Small sample:** n=16 makes validation and coefficients sensitive to individual conditions.
- **Related observations:** nested subsets reuse images and are not independent experimental units.
- **Reconstruction-derived reference:** the 50-image FAST_DENSE mesh is not scanner ground truth; its defects propagate into targets.
- **ICP sensitivity:** seven candidates have caution fitness (0.80–0.95). They are retained and separately audited.
- **Profile-specific geometry:** FAST_DENSE and FULL_REFERENCE meshes use different depth settings and cannot be pooled.
- **Feature collinearity:** count, reprojection, observation ratios, and overlap summaries correlate; small families reduce but do not eliminate unstable attribution.
- **Incomplete coverage representation:** global graph summaries do not encode where missing views occur, as the central-gap condition demonstrates.
- **Sampling approximation:** deterministic nearest-sample distances approximate exact surface distance, although self-check noise is effectively zero.
- **Misleading fit statistics:** R² and training fit would reward leverage and dependence at n=16, so out-of-fold errors are emphasized.
- **Complex ML is unjustified:** nonlinear high-capacity models and deep learning would be underconstrained and prone to memorization.
- **Exploratory status:** early-SfM improvement disappears under selection-family and pass-QC sensitivity.

Strong validation requires multiple artifacts and acquisition structures, artifact-level holdout, and preferably external geometric references.

