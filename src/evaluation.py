import math
import time
from typing import Callable
from pathlib import Path
from collections.abc import Iterable
from functools import partial
import numpy as np
import pandas as pd
import PIL.Image
from tqdm.contrib.concurrent import process_map

from src.data import Result, ResultWithComparison

def get_iou(pred: tuple[int, int, int], gt: tuple[int, int, int]) -> float:
    """IoU of two circles, each specified as (cy, cx, r)."""
    pred_cy, pred_cx, pred_r = pred
    gt_cy, gt_cx, gt_r = gt

    d = math.hypot(pred_cy - gt_cy, pred_cx - gt_cx)
    r1, r2 = max(pred_r, gt_r), min(pred_r, gt_r)

    if d >= pred_r + gt_r:
        return 0.0
    elif d <= r1 - r2:
        return math.pi * (r2**2) / (math.pi * (r1**2))

    # https://mathworld.wolfram.com/Circle-CircleIntersection.html
    d1 = (r1**2 - r2**2 + d**2) / (2 * d)
    d2 = d - d1
    
    # Avoid math domain error due to floating point precision issues
    val_to_sqrt = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    val_to_sqrt = max(0, val_to_sqrt)
    t = 0.5 * math.sqrt(val_to_sqrt)
    
    # Safe acos
    acos_arg1 = min(1.0, max(-1.0, d2 / r2))
    acos_arg2 = min(1.0, max(-1.0, d1 / r1))

    intersection_area = r2**2 * math.acos(acos_arg1) + r1**2 * math.acos(acos_arg2) - t
    union_area = math.pi * (pred_r**2 + gt_r**2) - intersection_area
    return intersection_area / union_area

def run_algorithm(
    img_path: Path,
    img_dir: Path,
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]],
) -> Result:
    """Runs a given algorithm on a single image and measures time."""
    pil_img = PIL.Image.open(img_path)
    start_time = time.perf_counter()
    cy, cx, r = algo(pil_img)
    duration = time.perf_counter() - start_time
    
    return {
        "path": str(img_path.relative_to(img_dir)),
        "h": pil_img.height,
        "w": pil_img.width,
        "cy": cy,
        "cx": cx,
        "r": r,
        "duration": round(duration, 4),
    }

def run_and_compare(
    img_path: Path, 
    img_dir: Path,
    ground_truths: dict[Path, dict],
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]]
) -> ResultWithComparison:
    """Runs algorithm and compares with ground truth if available."""
    pred = run_algorithm(img_path, img_dir, algo=algo)
    
    if img_path not in ground_truths:
        return {
            **pred,
            "c_dist": float("nan"),
            "r_diff": float("nan"),
            "iou": float("nan"),
        }
        
    gt = ground_truths[img_path]
    c_dist = np.hypot(pred["cy"] - gt["cy"], pred["cx"] - gt["cx"]) / gt["w"]
    r_diff = (pred["r"] - gt["r"]) / gt["r"]
    iou = get_iou((pred["cy"], pred["cx"], pred["r"]), (gt["cy"], gt["cx"], gt["r"]))
    
    return {
        **pred,
        "c_dist": round(float(c_dist), 4),
        "r_diff": round(float(r_diff), 4),
        "iou": round(float(iou), 4),
    }

def run_all(
    img_paths: Iterable[Path],
    img_dir: Path,
    ground_truths: dict[Path, dict],
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]],
    num_workers: int = 2,
) -> list[ResultWithComparison]:
    """Runs an algorithm on multiple images in parallel."""
    worker_func = partial(run_and_compare, img_dir=img_dir, ground_truths=ground_truths, algo=algo)
    return process_map(worker_func, img_paths, max_workers=num_workers)

def calculate_metrics(results_df: pd.DataFrame) -> tuple[float, float]:
    """Calculates success rate (IoU >= 0.85) and accuracy metrics."""
    success_rate = (results_df["iou"] >= 0.85).mean()
    accuracy = results_df["iou"].clip(0.85, 0.98).mean()
    return success_rate, accuracy
