# MTS

Experimental implementation of the **Mahalanobis-Taguchi System (MTS)** for binary classification and feature selection.

The project currently provides a scikit-learn-style classifier interface while the internal API and full scikit-learn integration are still under development.

> **Status:** experimental / alpha release

## Overview

The Mahalanobis-Taguchi System builds a reference space from observations considered **normal**, evaluates the separation of abnormal observations from that space, selects useful features using a Taguchi orthogonal array, and classifies new observations using their Mahalanobis Distance.

The current implementation follows this workflow:

```text
training data
    |
    v
split normal / abnormal
    |
    v
validation of the initial Mahalanobis space
    |
    v
Taguchi orthogonal-array optimization
    |
    v
S/N ratio comparison
    |
    v
feature selection (Delta S/N > 0)
    |
    v
optimized Mahalanobis Unit Space
    |
    v
chi-square classification threshold
    |
    v
prediction
```

## Current features

- Mahalanobis Unit Space construction
- Mahalanobis Distance calculation
- S/N ratio calculation
- MTS validation stage
- Taguchi orthogonal-array feature selection
- Feature selection using positive Delta S/N
- Chi-square-based classification threshold
- Binary `fit()` / `predict()` interface
- Initial scikit-learn-style estimator structure

Planned future work includes additional optimization strategies such as Bee Algorithm and Ant Colony Optimization, as well as fuller scikit-learn integration.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/b0nba/MTS.git
cd MTS
pip install -e .
```

## Basic usage

In the current implementation:

- `1` represents the **normal** class
- `0` represents the **abnormal** class

Example:

```python
import numpy as np

from mts import MTS


# Columns:
# 0 - temperature
# 1 - pressure
# 2 - vibration
# 3 - current

normal_data = np.array([
    [50.2, 100.5, 2.1, 10.0],
    [49.8,  99.7, 1.9, 10.4],
    [50.6, 101.2, 2.2,  9.8],
    [49.4, 100.8, 2.0, 10.1],
    [50.1,  98.9, 2.3,  9.7],
    [49.7, 101.0, 1.8, 10.3],
    [50.4,  99.5, 2.0,  9.9],
    [49.5, 100.2, 2.4, 10.2],
    [50.3, 100.7, 1.7,  9.6],
    [49.9,  99.1, 2.1, 10.5],
    [50.7, 100.0, 1.9, 10.0],
    [49.6, 101.3, 2.2,  9.8],
])

abnormal_data = np.array([
    [56.0, 108.0, 4.5, 13.0],
    [44.0,  92.0, 0.7,  7.5],
    [55.0,  95.0, 4.0, 12.5],
    [45.0, 109.0, 0.9,  7.0],
    [58.0, 103.0, 5.2, 13.5],
    [43.0,  96.0, 0.6,  8.0],
    [54.0, 110.0, 4.8, 11.8],
    [46.0,  91.0, 1.0,  7.2],
])

X_train = np.vstack([normal_data, abnormal_data])

y_train = np.array([
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0,
    0, 0, 0, 0
])

oa_design = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0],
])

model = MTS(
    opt=oa_design,
    alpha=0.05
)

model.fit(X_train, y_train)

X_test = np.array([
    [50.0, 100.1, 2.0, 10.1],
    [49.6,  99.8, 2.2,  9.9],
    [56.0, 107.0, 4.3, 12.8],
    [44.5,  93.0, 0.8,  7.4],
])

predictions = model.predict(X_test)

print(predictions)
```

Expected output:

```text
[1 1 0 0]
```

## Feature selection

The current implementation uses a Taguchi orthogonal array. The user must provide an orthogonal array design explicitly. This package does not currently generate orthogonal arrays automatically; external tools such as `oapackage` may be useful for constructing an appropriate design.

For each feature, the average S/N ratio is compared when the feature is included and excluded.

```text
Delta S/N = mean(S/N included) - mean(S/N excluded)
```

A feature is retained when:

```text
Delta S/N > 0
```

Therefore:

- positive Delta S/N -> feature is retained
- negative or zero Delta S/N -> feature is removed

The optimizer API is expected to be generalized in future releases so alternative feature-selection algorithms can be used.

## Classification threshold

Feature selection and classification-threshold determination are separate stages.

After feature selection, the final Mahalanobis Unit Space is constructed using only selected normal features.

The current implementation uses a chi-square-based classification threshold.

The Mahalanobis Distance returned by this package is normalized by the number of selected features:

```text
MD = D^2 / p
```

where:

- `D^2` is the squared Mahalanobis Distance
- `p` is the number of selected features

The decision threshold is calculated as:

```text
chi2.ppf(1 - alpha, df=p) / p
```

The default value is:

```python
alpha = 0.05
```

Prediction follows the rule:

```text
MD <= threshold  -> normal   (1)
MD >  threshold  -> abnormal (0)
```

`alpha` should satisfy:

```text
0 < alpha < 1
```

## Validation stage

Before optimization, the implementation evaluates the initial Mahalanobis space using the normal and abnormal observations.

The current validation output includes:

- mean Mahalanobis Distance of the normal space
- S/N ratio of the normal space
- mean Mahalanobis Distance of abnormal observations
- separation between abnormal and normal mean MD

The current alpha release prints these diagnostics during `fit()`.

## Project structure

```text
src/
└── mts/
    ├── __init__.py
    ├── mts.py
    ├── _math.py
    ├── _threshold.py
    ├── _validation.py
    │
    └── optimizers/
        ├── __init__.py
        └── taguchi.py
```

Responsibilities:

- `mts.py` - classifier orchestration (`fit`, `predict`)
- `_math.py` - Mahalanobis-space and S/N calculations
- `_validation.py` - MTS validation stage
- `_threshold.py` - classification-threshold calculation
- `optimizers/` - placeholder for future interchangeable optimization strategies

## scikit-learn integration

The current classifier inherits from scikit-learn's `BaseEstimator` and `ClassifierMixin` and follows a `fit()` / `predict()` interface.

Full scikit-learn estimator compatibility, pipeline support, estimator checks, and a generalized optimizer API are planned for future releases.

The current release should therefore be treated as an experimental implementation rather than a fully scikit-learn-compatible estimator.

## Limitations

The current release intentionally focuses on a minimal implementation.

Important limitations include:

- binary classification only
- class labels are currently fixed to `1 = normal` and `0 = abnormal`
- Taguchi/OA is currently the only implemented feature-selection approach
- the user must provide an appropriate orthogonal-array design
- singular or poorly conditioned correlation matrices are not yet handled automatically
- constant features may lead to invalid standardization
- input validation is currently minimal
- full scikit-learn compatibility is not yet guaranteed

## Roadmap
Highest priority:
- more detailed testing, numerical, statistical edge-cases testing
  
Further planned development includes:
- modular optimizer API
- Bee Algorithm optimization
- Ant Colony Optimization
- additional threshold-selection strategies
- improved numerical stability
- improved input validation
- full scikit-learn integration
- expanded test coverage and documentation

## License

See the `LICENSE` file included in this repository.
