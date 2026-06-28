import math
from pathlib import Path
from typing import Callable, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import PIL.ImageDraw
import ipywidgets
from IPython.display import display

from src.data import Result
from src.evaluation import run_algorithm, run_and_compare

def show_result(
    img_dir: Path,
    result: Result, 
    gt: Result | None = None, 
    max_height: int = 512
) -> PIL.Image.Image:
    """Visualizes the algorithm's result on an image, drawing the prediction and the ground truth."""
    # print(result)
    pil_img = PIL.Image.open(img_dir / result["path"])
    draw = PIL.ImageDraw.Draw(pil_img)

    # Draw the ground truth in red, in the background, with a small circle at the center
    if gt:
        xy = (gt["cx"], gt["cy"])
        draw.circle(xy=xy, radius=gt["r"], outline="red", width=2)
        draw.circle(xy=xy, radius=30, outline="red", width=2)

    # Draw our prediction in blue
    xy = (result["cx"], result["cy"])
    draw.circle(xy=xy, radius=result["r"], outline="blue", width=3)
    draw.circle(xy=xy, radius=30, outline="blue", width=3)

    downscale = max(1, math.ceil(pil_img.height / max_height))
    return pil_img.reduce(downscale)

def show_results(
    img_paths: Sequence[Path] | pd.Index,
    img_dir: Path,
    ground_truths: dict[Path, dict],
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]],
    init_i: int = 0,
) -> None:
    """Interactive widget to browse through results on a set of images."""
    if isinstance(img_paths, pd.Index):
        img_paths = list(img_paths)

    def f(i: int) -> None:
        print(f"Image {i + 1}/{len(img_paths)}: {img_paths[i].name}")
        if img_paths[i] in ground_truths:
            result = run_and_compare(img_paths[i], img_dir, ground_truths, algo=algo)
            display(show_result(img_dir, result, ground_truths[img_paths[i]]))
        else:
            result = run_algorithm(img_paths[i], img_dir, algo=algo)
            display(show_result(img_dir, result))

    ipywidgets.interact(
        f, i=ipywidgets.IntSlider(min=0, max=len(img_paths) - 1, value=init_i)
    )

def plot_history(history: dict[str, list[float]]) -> None:
    """Plots the learning curves (Loss) for the deep learning model."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
