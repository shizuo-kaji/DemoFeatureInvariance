#!/usr/bin/env python3
"""
Generate a CSV dataset of triangles given three points per sample.

Columns:
  x1,y1,x2,y2,x3,y3,u,v,w

Definitions / assumptions:
  - u: sum of centroid coordinates = (x1+x2+x3)/3 + (y1+y2+y3)/3
  - v: triangle area (absolute, via shoelace)
  - w: ratio of inradius to circumradius (r/R)

Triangles with (near-)zero area are rejected and resampled.
"""
import argparse
import csv
import math
import random
from typing import Tuple, Optional


def area_triangle(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    # Shoelace formula
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)


def centroid_sum(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    cx = (x1 + x2 + x3) / 3.0
    cy = (y1 + y2 + y3) / 3.0
    return cx + cy


def dist(xa: float, ya: float, xb: float, yb: float) -> float:
    return math.hypot(xa - xb, ya - yb)


def inradius_circumradius_ratio(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> Optional[float]:
    """
    Compute the ratio of inradius (r) to circumradius (R) for a triangle.

    r = Area / s  (where s is the semi-perimeter)
    R = (a * b * c) / (4 * Area)
    r / R = 4 * Area^2 / (s * a * b * c)
    """
    # Side lengths
    a = dist(x2, y2, x3, y3)
    b = dist(x1, y1, x3, y3)
    c = dist(x1, y1, x2, y2)

    # If any side is zero (duplicate points) return None
    if a == 0 or b == 0 or c == 0:
        return None

    area = area_triangle(x1, y1, x2, y2, x3, y3)
    if area == 0:
        return None

    s = (a + b + c) / 2.0  # semi-perimeter

    # r/R = 4 * Area^2 / (s * a * b * c)
    return (4.0 * area * area) / (s * a * b * c)


def generate_sample(xmin: float, xmax: float, ymin: float, ymax: float, min_area: float, max_attempts: int=100) -> Tuple[float, float, float, float, float, float]:
    for _ in range(max_attempts):
        x1 = random.uniform(xmin, xmax)
        y1 = random.uniform(ymin, ymax)
        x2 = random.uniform(xmin, xmax)
        y2 = random.uniform(ymin, ymax)
        x3 = random.uniform(xmin, xmax)
        y3 = random.uniform(ymin, ymax)

        if area_triangle(x1, y1, x2, y2, x3, y3) >= min_area:
            return x1, y1, x2, y2, x3, y3

    # If we failed to find a non-degenerate triangle, return the last one anyway
    return x1, y1, x2, y2, x3, y3


def make_dataset(n: int, output: str, seed: Optional[int], xmin: float, xmax: float, ymin: float, ymax: float, min_area: float, noise: float) -> None:
    if seed is not None:
        random.seed(seed)

    header = ["x1", "y1", "x2", "y2", "x3", "y3", "u", "v", "w"]

    with open(output, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(n):
            x1, y1, x2, y2, x3, y3 = generate_sample(xmin, xmax, ymin, ymax, min_area)
            u = centroid_sum(x1, y1, x2, y2, x3, y3)
            v = area_triangle(x1, y1, x2, y2, x3, y3)
            w = inradius_circumradius_ratio(x1, y1, x2, y2, x3, y3)
            # If ratio computation failed (shouldn't for valid area), set NaN
            if w is None:
                w = float('nan')
            # Add noise to u, v, w
            u += random.gauss(0, noise)
            v += random.gauss(0, noise)
            w += random.gauss(0, noise)

            writer.writerow([f"{x1:.8f}", f"{y1:.8f}", f"{x2:.8f}", f"{y2:.8f}", f"{x3:.8f}", f"{y3:.8f}", f"{u:.8f}", f"{v:.8f}", f"{w:.8f}"])


def parse_args():
    p = argparse.ArgumentParser(description="Generate triangle dataset CSV")
    p.add_argument("--n", type=int, default=1000, help="number of samples")
    p.add_argument("--output", type=str, default="triangle_dataset.csv", help="output CSV file path")
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    p.add_argument("--xmin", type=float, default=-1.0, help="min x coordinate")
    p.add_argument("--xmax", type=float, default=1.0, help="max x coordinate")
    p.add_argument("--ymin", type=float, default=-1.0, help="min y coordinate")
    p.add_argument("--ymax", type=float, default=1.0, help="max y coordinate")
    p.add_argument("--min-area", type=float, default=1e-6, help="minimum allowed triangle area (reject smaller)")
    p.add_argument("--noise", type=float, default=0.1, help="standard deviation of Gaussian noise added to u,v,w")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_dataset(args.n, args.output, args.seed, args.xmin, args.xmax, args.ymin, args.ymax, args.min_area, args.noise)
