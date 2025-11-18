# Demo for data analysis: importance of representation choice in regression tasks

This is prepared for my class "Interdisciplinary Study S1", Autumn 2025.

## Purpose of this demo

This demo is designed to illustrate a simple but important point in applied machine learning using a synthetic dataset based on 2D triangles.

If you would like to tackle the task independently without any hints:

- Please refer only to the training dataset **triangle_dataset_train.csv** and the test dataset **triangle_dataset_test.csv**.
- Your goal is to perform regression using the six columns **(x1, y1, x2, y2, x3, y3)** as explanatory variables to predict the target variables **u, v, w**.
- Build your own model and evaluate it using the R^2 score on the test dataset.

## Key idea: representation and task-relevant information
The dataset is constructed as follows: each data point represents a triangle in 2D defined by three points p1, p2, p3 with coordinates (x1,y1), (x2,y2), (x3,y3). The target variables are derived features of the triangle:
- u: sum of centroid coordinates = (x_centroid + y_centroid)
- v: area of the triangle
- w: sum of squared angles at the three vertices (angles in radians, squared and summed)

The key learning point of this demo is how the choice of input representation (features) affects regression performance depending on the target variable.
- Raw data often contains multiple kinds of information (different "aspects").
In this example, the six coordinates (x1,y1,x2,y2,x3,y3) encode absolute position, orientation, scale, and the triangle's intrinsic shape.
- The prediction target (u, v or w) depends only on a subset of those aspects (or they have "invariance"). For instance, u (centroid sum) depends on position, v (area) depends on shape *and* scale, and w (angle-based) depends on shape but not scale.
- Domain knowledge lets us build representations that emphasize the target-relevant aspects. The `--preprocess congruent` and `--preprocess similar` options are simple examples: they remove nuisance variability (translation, rotation, or also scale) so the learner sees a representation closer to the information the target depends on.
- If the preprocessing preserves the necessary information (the assumption is correct), learning becomes easier and model performance improves. If preprocessing removes information the target actually needs, performance will degrade — this trade-off is visible in the experiment results.

This idea connects to the manifold assumption: although raw inputs live in a high-dimensional ambient space (here R^6), the data relevant for the task often lies near a lower-dimensional manifold (for instance, triangles considered up to congruence or similarity form a lower-dimensional set). Effective preprocessing or feature design aims to map raw inputs closer to this task-specific manifold, reducing irrelevant variation and making the learning problem easier.

Use this demo to experiment with different representations and to see how domain-driven abstraction affects regression quality.


## Demo on Google Colab

Open [this notebook](colab_regression_demo.ipynb) in Google Colab to run the demo without local setup.


## Local setup

Install required Python packages listed in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Triangle dataset generator

This simple script generates a CSV file where each row is a sample consisting of three 2D points (a triangle) and three derived features.

Columns:
- x_1,y_1,x_2,y_2,x_3,y_3 : coordinates of the three points
- u : sum of centroid coordinates = (x_centroid + y_centroid)
- v : area of the triangle (absolute, via shoelace formula)
- w : sum of squared angles at the three vertices (angles in radians, squared and summed)


Usage example:

```bash
python generate_triangle_dataset.py --n 1000 --output triangle_dataset.csv --seed 42 --xmin -10 --xmax 10 --ymin -10 --ymax 10
```

This writes `triangle_dataset.csv` with 1000 non-degenerate triangles sampled uniformly in the box [-10,10] x [-10,10]. Triangles with area smaller than `--min-area` are rejected and resampled.

### Regression models

There's a demonstration script that trains several regressors to predict the derived features (u, v, w) from the six point coordinates.

File: `regression/regression_demo.py`

Basic usage (single CSV, internal train/test split):

```bash
python regression/regression_demo.py --csv triangle_dataset.csv --out regression_demo_out --test-size 0.2 --seed 0
```

### Use a separate test CSV

If you'd like to provide a separate test dataset (for example `triangle_dataset_train.csv` and `triangle_dataset_test.csv`), use the `--csv-test` option. When present, the script will train on `--csv` and evaluate on `--csv-test` (no internal split is performed):

```bash
python regression/regression_demo.py \
	--csv triangle_dataset_train.csv \
	--csv-test triangle_dataset_test.csv \
	--out regression_demo_out \
	--seed 0
```

### Output

- `regression_demo_out/regression_results.csv`: table of MSE, MAE, and R2 for each (target, regressor).
- `regression_demo_out/pred_vs_true_{target}_{regressor}.png`: scatter plots of predicted vs true values with a y=x reference line.


## Preprocessing options

The demo supports an optional preprocessing step that normalizes triangle coordinates before training/evaluation.

- `none` (default): no preprocessing.
- `congruent`: translate and rotate each triangle so that p1 becomes (0,0), p2 lies on the x-axis, and p3 has non-negative y (i.e., enforce a canonical congruent placement).
- `similar`: apply a similarity transform (translation, rotation, uniform scaling) so that p1 becomes (0,0), p2 becomes (1,0), and p3 has non-negative y (canonical placement up to similarity).
- `length`: transform each triangle into its three edge lengths (l_a, l_b, l_c). This representation removes absolute position and orientation while preserving scale; it's useful when the target depends on scale/size (for example, area).
- `angle`: transform each triangle into two corner angles (angles at p1 and p2). This removes position, orientation, and scale information and keeps only intrinsic shape information; it's useful when the target depends only on shape (for example, angle-based targets like `w`).

Example (similar preprocessing):

```bash
python regression/regression_demo.py 	--csv triangle_dataset_train.csv --csv-test triangle_dataset_test.csv --preprocess similar --out regression_demo_out/similar
```

## Discussion of results

This file summarises the experimental outputs produced by `regression/regression_demo.py` and reflects the educational purpose of the demo: to show how different input representations (preprocessing/feature choices) affect regression performance for different targets.

Location of outputs
- `regression/regression_demo_out/` contains one subfolder per preprocessing mode used in the experiments. Each subfolder includes:
  - `regression_results.csv` — metrics table (target, regressor, MSE, MAE, R2).
  - `pred_vs_true_{target}_{regressor}.png` — scatter plots comparing predictions vs ground truth.

Modes present (as of this run)
- none — raw coordinates (x1,y1,x2,y2,x3,y3)
- congruent — canonical congruent placement (translation + rotation + optional flip)
- similar — canonical similarity placement (translation + rotation + uniform scaling + flip)
- length — features = three edge lengths (l_a,l_b,l_c)
- angle — features = two corner angles (angles at p1 and p2, in radians)

Key numeric highlights (best-performing regressor per target in each mode, chosen by highest R2 in the saved `regression_results.csv`):

- Mode: none (raw coordinates)
  - u: Linear R2 = 1.000 (perfect fit) — centroid-sum `u` is linear in coordinates so a linear regressor recovers it exactly.
  - v: MLP R2 ≈ 0.601 — a non-linear model gave the best performance for area `v` on raw coordinates.
  - w: KNN R2 ≈ 0.142 (small positive) — angle-sum-squared `w` is moderately difficult from raw coords.

- Mode: congruent (remove translation+rotation)
  - u: SVR R2 ≈ 0.009 (very poor) — centroid-sum depends on absolute position, which congruent placement removes.
  - v: MLP R2 ≈ 0.992 (excellent) — removing translation/rotation helped predict area `v` much better.
  - w: MLP R2 ≈ 0.757 — angles and intrinsic shape can be predicted well after removing nuisance orientation.

- Mode: similar (remove translation+rotation+scale)
  - u: Ridge/Linear R2 ≈ 0.004 (very poor) — as expected, scale/position are removed so `u` can't be predicted.
  - v: RandomForest R2 ≈ 0.426 — removing scale harms area prediction; models cannot recover absolute scale from similarity-normalized inputs.
  - w: RandomForest R2 ≈ 0.621 — shape-only representation still allows fairly good prediction of angle-based `w`.

- Mode: length (edge lengths l_a,l_b,l_c)
  - u: Ridge R2 ≈ -0.024 (poor) — lengths remove position, so centroid sum `u` is not recoverable.
  - v: MLP R2 ≈ 0.903 (very good) — edge lengths retain scale information and strongly determine area.
  - w: MLP R2 ≈ 0.612 — lengths provide enough shape information to predict angle-based measures moderately well.

- Mode: angle (angles at two corners)
  - u: SVR R2 ≈ 0.000 (near 0) — angles discard position/scale so `u` is unrecoverable.
  - v: MLP R2 ≈ 0.317 — angles discard scale, so area prediction suffers.
  - w: MLP R2 ≈ 0.972 (excellent) — angle features give near-perfect prediction for `w` (which is computed from angles).

Interpretation and reflections

- Representation matters: these experiments show the expected trade-offs.
  - If the target depends on absolute position (u), representations that remove position destroy predictive signal and performance collapses.
  - If the target depends on scale (v), representations that remove scale (similar, angle) reduce performance; representations preserving scale (raw coords, length) perform better.
  - If the target depends only on intrinsic shape (w), shape-preserving representations that remove nuisance variability (congruent, angle) make learning easier — angle features are especially strong for `w` because `w` is directly derived from angles.

- Model choice interacts with representation:
  - Linear models excel when the mapping is linear in the features (e.g., `u` from raw coordinates).
  - Non-linear models (MLP, RandomForest) shine when the mapping is non-linear (e.g., area from lengths or raw coordinates).

- Practical lesson for applied ML:
  - Use domain knowledge to remove nuisance factors (translation, rotation, scale) only when those factors are irrelevant to the target.
  - When in doubt, run small controlled experiments (like these) comparing representations — the correct representation often yields much better performance than simply throwing more model capacity at raw data.
