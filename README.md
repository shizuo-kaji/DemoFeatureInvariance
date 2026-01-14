# Regression & Representation: Educational Demo

**Course:** Interdisciplinary Study S1, Autumn 2025

This project demonstrates how **input representation** (feature engineering) critically impacts machine learning performance. You will explore this concept using a simple, interpretable geometric dataset: 2D triangles.

## Learning Objectives

By working through this demo, you will understand:
1.  **Feature Representation**: Why raw data (e.g., coordinates) isn't always the best input for a model.
2.  **Invariance**: How removing "nuisance" variations (like rotation or translation) can simplify learning.
3.  **Inductive Bias**: How different models (Linear, Trees, Neural Nets) have different strengths depending on the data structure.
4.  **Target Dependence**: A representation good for predicting *area* might be bad for predicting *position*.

---

## The Task

You are given a dataset of triangles. Each triangle is defined by 3 points in 2D space: $p_1(x_1, y_1)$, $p_2(x_2, y_2)$, $p_3(x_3, y_3)$.

Your goal is to predict three different target variables from these coordinates:
1.  **$u$ (Position)**: The sum of the centroid coordinates $(x_c + y_c)$. Depends on **absolute position**.
2.  **$v$ (Area)**: The area of the triangle. Depends on **shape and scale** (invariant to position/rotation).
3.  **$w$ (Shape)**: The sum of squared angles at the vertices. Depends only on **intrinsic shape** (invariant to position, rotation, and scale).

### The Challenge
Before running the code, ask yourself:
*   Can a Linear Regression model predict the Area ($v$) from raw coordinates?
*   If we remove all information about position (center the triangles), what happens to our ability to predict $u$?
*   Which representation is best for predicting angles ($w$)?

---

## Quick Start

## Run on Google Colab
[Open in Colab](colab_regression_demo.ipynb) to run without local installation.

## Run locally on your PC

### 1. Setup
(Optional) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Basic Demo
Train models using **raw coordinates** (no preprocessing).
```bash
python regression_demo.py --csv triangle_dataset_train.csv --csv-test triangle_dataset_test.csv --out regression_demo_out/raw
```
*   Check the results in `regression_demo_out/raw/regression_results.csv`.
*   Look at the plots in `regression_demo_out/raw/`.

---

## Experiments: Exploring Representations

The script supports a `--preprocess` flag to transform the input data before training. Run these experiments to see how representation changes performance.

### Experiment A: Removing Position & Rotation (`--preprocess congruent`)
This centers every triangle at the origin and aligns it with the x-axis.
```bash
python regression_demo.py --csv triangle_dataset_train.csv --csv-test triangle_dataset_test.csv --preprocess congruent --out regression_demo_out/congruent
```
*   **Hypothesis**: Prediction for $u$ (position) should fail. Prediction for $v$ (area) and $w$ (angles) should improve or stay good.

### Experiment B: Removing Scale (`--preprocess similar`)
This normalizes the size of every triangle (making the base length 1), in addition to centering and rotating.
```bash
python regression_demo.py --csv triangle_dataset_train.csv --csv-test triangle_dataset_test.csv --preprocess similar --out regression_demo_out/similar
```
*   **Hypothesis**: Prediction for $v$ (area) should suffer because we destroyed size information. Prediction for $w$ (angles) should remain good.

### Experiment C: Explicit Features (`--preprocess length` and `--preprocess angle`)
Instead of coordinates, let's feed the model explicit geometric features.

**Edge Lengths ($l_a, l_b, l_c$):**
```bash
python regression_demo.py --csv triangle_dataset_train.csv --csv-test triangle_dataset_test.csv --preprocess length --out regression_demo_out/length
```

**Corner Angles ($\alpha, \beta$):**
```bash
python regression_demo.py --csv triangle_dataset_train.csv --csv-test triangle_dataset_test.csv --preprocess angle --out regression_demo_out/angle
```

---

## Analysis of Results

Below is a summary of what you typically observe. Use this to verify your understanding.

| Target | Best Representation | Why? |
| :--- | :--- | :--- |
| **$u$ (Centroid)** | `none` (Raw Coords) | $u$ is a linear function of coordinates. Any preprocessing that removes position destroys the signal. |
| **$v$ (Area)** | `congruent` or `length` | Area is independent of position/rotation but depends on scale. `congruent` simplifies the task by fixing position. `length` captures scale directly. `similar` and `angle` fail because they remove scale. |
| **$w$ (Angles)** | `angle` | The target is directly derived from angles. `similar` and `congruent` also work well because they expose the shape clearly. |

### Key Takeaways for Applied ML
1.  **Don't blindly preprocess**: Removing "noise" (like position) is only good if your target doesn't depend on it.
2.  **Domain Knowledge is Power**: If you know your target depends only on shape, use shape-based features (angles/ratios) to make the model's job easier.
3.  **Model vs. Data**: A simple model (Linear Regression) with the right features (squared terms for area, or explicit lengths) can outperform a complex model (Neural Net) on raw data.

---

## Project Structure

*   `generate_triangle_dataset.py`: Generates the synthetic data.
*   `regression_demo.py`: The main training and evaluation script.
    *   Look at `_apply_preprocess_row` to see how the geometric transformations are implemented.
*   `triangle_dataset_train.csv` / `triangle_dataset_test.csv`: The default datasets.
