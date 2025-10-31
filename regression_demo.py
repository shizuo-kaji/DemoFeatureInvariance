#!/usr/bin/env python3
"""
Educational regression demo for triangle dataset.

Input CSV columns: x1,y1,x2,y2,x3,y3,u,v,w
This script trains and compares multiple regressors to predict u, v, and w from the six coordinates.

It prints a results table (MSE, MAE, R2) and saves simple predicted vs actual scatter plots into an output folder.
"""
import argparse
import os
from typing import List
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


REGRESSORS = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
    "SVR": SVR(C=1.0, kernel="rbf"),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0),
}


# ---------- Geometry helpers for preprocessing and target recomputation ----------
def _dist(xa: float, ya: float, xb: float, yb: float) -> float:
    return np.hypot(xa - xb, ya - yb)


def _area_triangle_row(row) -> float:
    x1, y1, x2, y2, x3, y3 = (row["x1"], row["y1"], row["x2"], row["y2"], row["x3"], row["y3"])
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)


def _centroid_sum_row(row) -> float:
    cx = (row["x1"] + row["x2"] + row["x3"]) / 3.0
    cy = (row["y1"] + row["y2"] + row["y3"]) / 3.0
    return cx + cy


def _sum_squared_angles_row(row) -> float:
    x1, y1, x2, y2, x3, y3 = (row["x1"], row["y1"], row["x2"], row["y2"], row["x3"], row["y3"])
    a = _dist(x2, y2, x3, y3)
    b = _dist(x1, y1, x3, y3)
    c = _dist(x1, y1, x2, y2)
    # handle degenerate
    if a == 0 or b == 0 or c == 0:
        return float("nan")

    def _safe_acos(val: float) -> float:
        return math.acos(max(-1.0, min(1.0, val)))

    A = _safe_acos((b*b + c*c - a*a) / (2*b*c))
    B = _safe_acos((a*a + c*c - b*b) / (2*a*c))
    C = _safe_acos((a*a + b*b - c*c) / (2*a*b))
    return A*A + B*B + C*C


def _recompute_targets_from_coords(df_coords: pd.DataFrame, target: str) -> pd.Series:
    # df_coords has columns x1,y1,x2,y2,x3,y3
    if target == "u":
        return df_coords.apply(_centroid_sum_row, axis=1)
    elif target == "v":
        return df_coords.apply(_area_triangle_row, axis=1)
    elif target == "w":
        return df_coords.apply(_sum_squared_angles_row, axis=1)
    else:
        raise ValueError(f"Unknown target for recompute: {target}")


def _apply_preprocess_row(row, mode: str):
    # return transformed coordinate tuple (x1,y1,x2,y2,x3,y3) as floats
    x1, y1, x2, y2, x3, y3 = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), float(row["x3"]), float(row["y3"]))

    # translate so p1 at origin
    tx2 = x2 - x1
    ty2 = y2 - y1
    tx3 = x3 - x1
    ty3 = y3 - y1

    if mode == "congruent":
        # rotate so p2 on x-axis
        theta = math.atan2(ty2, tx2)
        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)

        def rot(x, y):
            xr = x * cos_t - y * sin_t
            yr = x * sin_t + y * cos_t
            return xr, yr

        r2x, r2y = rot(tx2, ty2)
        r3x, r3y = rot(tx3, ty3)

        # enforce y3 >= 0 by flipping across x-axis if necessary
        if r3y < 0:
            r2y = -r2y
            r3y = -r3y

        return (0.0, 0.0, float(r2x), float(r2y), float(r3x), float(r3y))

    elif mode == "similar":
        # scale so p2 at (1,0) after rotation
        d = math.hypot(tx2, ty2)
        if d == 0:
            scale = 1.0
            theta = 0.0
        else:
            scale = 1.0 / d
            theta = math.atan2(ty2, tx2)

        cos_t = math.cos(-theta)
        sin_t = math.sin(-theta)

        def rot_scale(x, y):
            xr = (x * cos_t - y * sin_t) * scale
            yr = (x * sin_t + y * cos_t) * scale
            return xr, yr

        r2x, r2y = rot_scale(tx2, ty2)
        r3x, r3y = rot_scale(tx3, ty3)

        # numerical tolerance: r2x should be ~1, r2y ~0
        # enforce y3 >= 0
        if r3y < 0:
            r2y = -r2y
            r3y = -r3y

        return (0.0, 0.0, float(r2x), float(r2y), float(r3x), float(r3y))

    else:
        # none or unknown: return original
        return (x1, y1, x2, y2, x3, y3)


def _apply_preprocess_df(df_coords: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode is None or mode == "none":
        return df_coords.copy()

    cols = ["x1", "y1", "x2", "y2", "x3", "y3"]
    transformed = df_coords.apply(lambda r: pd.Series(_apply_preprocess_row(r, mode), index=cols), axis=1)
    return transformed


def _coords_to_length_df(df_coords: pd.DataFrame) -> pd.DataFrame:
    # sides: a = dist(p2,p3), b = dist(p1,p3), c = dist(p1,p2)
    def row_lengths(r):
        x1, y1, x2, y2, x3, y3 = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["x3"]), float(r["y3"]))
        a = np.hypot(x2 - x3, y2 - y3)
        b = np.hypot(x1 - x3, y1 - y3)
        c = np.hypot(x1 - x2, y1 - y2)
        return pd.Series({"l_a": a, "l_b": b, "l_c": c})

    return df_coords.apply(row_lengths, axis=1)


def _coords_to_angle_df(df_coords: pd.DataFrame) -> pd.DataFrame:
    # compute two angles: at p1 (A) and at p2 (B)
    def row_angles(r):
        x1, y1, x2, y2, x3, y3 = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["x3"]), float(r["y3"]))
        a = np.hypot(x2 - x3, y2 - y3)
        b = np.hypot(x1 - x3, y1 - y3)
        c = np.hypot(x1 - x2, y1 - y2)
        # handle degenerate cases
        if a == 0 or b == 0 or c == 0:
            return pd.Series({"ang1": float("nan"), "ang2": float("nan")})

        def safe_acos(val):
            return math.acos(max(-1.0, min(1.0, val)))

        A = safe_acos((b*b + c*c - a*a) / (2*b*c))
        B = safe_acos((a*a + c*c - b*b) / (2*a*c))
        return pd.Series({"ang1": A, "ang2": B})

    return df_coords.apply(row_angles, axis=1)


def _transform_features(df_coords: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a feature DataFrame for the given preprocess mode.
    For coordinate-based modes (congruent/similar/none) this returns x1..y3.
    For 'length' returns l_a,l_b,l_c. For 'angle' returns ang1,ang2.
    """
    if mode is None or mode == "none":
        return df_coords.copy()
    if mode in ("congruent", "similar"):
        return _apply_preprocess_df(df_coords, mode)
    if mode == "length":
        return _coords_to_length_df(df_coords)
    if mode == "angle":
        return _coords_to_angle_df(df_coords)
    # fallback
    return df_coords.copy()


# ---------- end geometry helpers ----------


def build_pipeline(regressor):
    # For some regressors scaling improves performance (SVR, MLP, KNN, linear models)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", regressor)
    ])


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, outpath: str, title: str):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=20, alpha=0.7)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], color="red", linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def run_demo(csv_path: str, out_dir: str, csv_test: str, test_size: float, random_seed: int, targets: List[str], regressors_to_use: List[str], preprocess: str = "none"):
    os.makedirs(out_dir, exist_ok=True)

    # Load training data
    df = pd.read_csv(csv_path)

    feature_cols = ["x1", "y1", "x2", "y2", "x3", "y3"]
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"Target column '{t}' not found in train CSV")

    # Drop rows with NaN in selected targets in train
    df = df.dropna(subset=targets + feature_cols)

    # Extract coordinate DataFrame for possible preprocessing
    df_coords = df[feature_cols]

    # Apply preprocessing / feature transform if requested
    df_coords_train = _transform_features(df_coords, preprocess)

    X = df_coords_train

    results = []

    if csv_test:
        # Use separate test CSV provided by user
        df_test = pd.read_csv(csv_test)
        # Validate test has necessary columns
        for t in targets:
            if t not in df_test.columns:
                raise ValueError(f"Target column '{t}' not found in test CSV")
        df_test = df_test.dropna(subset=targets + feature_cols)
        # preprocess/transform test coords as well
        df_test_coords = df_test[feature_cols]
        df_test_coords_trans = _transform_features(df_test_coords, preprocess)

        X_train = X
        X_test = df_test_coords_trans
    else:
        # Use train/test split from the single CSV
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_seed)

    # For each target, train each regressor separately and report metrics
    for target in targets:
        # Prepare target vectors depending on preprocessing mode
        if csv_test:
            y_train = df[target]
            y_test = df_test[target]
        else:
            y = df[target]
            y_train = y.loc[X_train.index]
            y_test = y.loc[X_test.index]

        for reg_name in regressors_to_use:
            base = REGRESSORS.get(reg_name)
            if base is None:
                print(f"Warning: regressor {reg_name} not recognized, skipping")
                continue

            pipe = build_pipeline(base)

            # Fit
            pipe.fit(X_train, y_train)

            mse, mae, r2, y_pred = evaluate_model(pipe, X_test, y_test)

            results.append({
                "target": target,
                "regressor": reg_name,
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
            })

            # Plot and save
            fname = f"pred_vs_true_{target}_{reg_name}.png"
            plot_pred_vs_true(y_test.values, y_pred, os.path.join(out_dir, fname), f"{target} - {reg_name}")

            print(f"Trained {reg_name} for target {target}: MSE={mse:.4g}, MAE={mae:.4g}, R2={r2:.4g}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["target", "MSE"]).reset_index(drop=True)
    results_df.to_csv(os.path.join(out_dir, "regression_results.csv"), index=False)
    print("\nSummary saved to:", os.path.join(out_dir, "regression_results.csv"))
    print("Plots saved to:", out_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Regression demo for triangle dataset")
    p.add_argument("--csv", type=str, default="triangle_dataset.csv", help="input CSV file path")
    p.add_argument("--csv-test", type=str, default="", help="optional: separate test CSV file path")
    p.add_argument("--out", type=str, default="regression_demo_out", help="output folder for results and plots")
    p.add_argument("--test-size", type=float, default=0.2, help="test set fraction")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--targets", type=str, default="u,v,w", help="comma-separated targets to predict (u,v,w)")
    p.add_argument("--regs", type=str, default="Linear,Ridge,DecisionTree,RandomForest,SVR,KNN,MLP", help="comma-separated regressors to run")
    p.add_argument("--preprocess", type=str, default="none", choices=["none", "congruent", "similar", "length", "angle"], help="optional preprocessing: none, congruent, similar, length, or angle")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    regs = [r.strip() for r in args.regs.split(",") if r.strip()]
    run_demo(args.csv, args.out, args.csv_test, args.test_size, args.seed, targets, regs, preprocess=args.preprocess)
