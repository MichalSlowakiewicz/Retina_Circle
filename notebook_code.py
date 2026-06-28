# Cell 2
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import cv2
import ipywidgets
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageDraw
from IPython.display import display
from tqdm.contrib.concurrent import process_map

# Cell 4
# added imports
from pathlib import Path
from typing import TypedDict, Callable
import json
import pandas as pd
import PIL
from PIL import Image
from collections.abc import Iterable, Sequence
import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from functools import partial
import csv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

# Cell 5
%%bash
mkdir -p data/
if [ ! -d "data/circle/train" ]; then
    # 50 MiB disk space
    wget -q 'https://www.mimuw.edu.pl/~mw290715/upload/circle.tar.gz' -O data/circle.tar.gz
    cd data/
    tar -xf circle.tar.gz
    cd -
fi

# Cell 6
IMG_DIR = Path("data/circle/")
train_set = sorted((IMG_DIR / "train").rglob("*.webp"))
val_set = sorted((IMG_DIR / "val").rglob("*.webp"))
assert len(train_set) == 1000 and len(val_set) == 100, (
    "Bad number of images in train/val sets."
)

# Cell 7
class Result(TypedDict):
    path: str  # Relative to IMG_DIR. Always an RGB webp image.
    h: int  # Image height.
    w: int  # Image width.
    cy: int  # Y coordinate of circle center.
    cx: int  # X coordinate of circle center.
    r: int  # Radius of the circle.
    duration: float  # Time taken to find the circle, in seconds.

# Cell 8
GROUND_TRUTHS: dict[Path, Result] = {}

for part in ("train", "val", "test"):
    if not (IMG_DIR / f"{part}.json").exists():
        continue
    with (IMG_DIR / f"{part}.json").open() as f:
        for result in json.load(f):
            GROUND_TRUTHS[IMG_DIR / result["path"]] = result

df = pd.DataFrame.from_dict(GROUND_TRUTHS, orient="index")
df["difficulty"] = df.index.map(lambda p: p.parent.name)
df.head()

# Cell 10
def run_algorithm(
    img_path: Path,
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]],
) -> Result:
    pil_img = PIL.Image.open(img_path)
    start_time = time.perf_counter()
    cy, cx, r = algo(pil_img)
    duration = time.perf_counter() - start_time
    return {
        "path": str(img_path.relative_to(IMG_DIR)),
        "h": pil_img.height,
        "w": pil_img.width,
        "cy": cy,
        "cx": cx,
        "r": r,
        "duration": round(duration, 4),
    }

# Cell 11
class ResultWithComparison(Result):
    c_dist: float  # Distance from predicted to GT center, as ratio of image width.
    r_diff: float  # Difference of predicted minus GT radius (possibly negative), as ratio of GT radius.
    iou: float  # Intersection over Union of predicted vs GT disc. Between 0 and 1.


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
    t = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    intersection_area = r2**2 * math.acos(d2 / r2) + r1**2 * math.acos(d1 / r1) - t
    union_area = math.pi * (pred_r**2 + gt_r**2) - intersection_area
    return intersection_area / union_area


def run_and_compare(
    img_path: Path, algo: Callable[[PIL.Image.Image], tuple[int, int, int]]
) -> ResultWithComparison:
    pred = run_algorithm(img_path, algo=algo)
    if img_path not in GROUND_TRUTHS:
        return {
            **pred,
            "c_dist": float("nan"),
            "r_diff": float("nan"),
            "iou": float("nan"),
        }
    gt = GROUND_TRUTHS[img_path]
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
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]],
    num_workers: int = 2,
) -> list[ResultWithComparison]:
    return process_map(
        partial(run_and_compare, algo=algo), img_paths, max_workers=num_workers
    )

# Cell 13
def show_result(
    result: Result, gt: Result | None = None, max_height: int = 512
) -> PIL.Image.Image:
    print(result)
    pil_img = PIL.Image.open(IMG_DIR / result["path"])
    draw = PIL.ImageDraw.Draw(pil_img)

    # Draw ground truth in red, behind, with a small circle at the center.
    if gt:
        xy = (gt["cx"], gt["cy"])
        draw.circle(xy=xy, radius=gt["r"], outline="red", width=2)
        draw.circle(xy=xy, radius=30, outline="red", width=2)

    # Draw this result in blue, with a small circle at the center.
    xy = (result["cx"], result["cy"])
    draw.circle(xy=xy, radius=result["r"], outline="blue", width=3)
    draw.circle(xy=xy, radius=30, outline="blue", width=3)

    downscale = max(1, math.ceil(pil_img.height / max_height))
    return pil_img.reduce(downscale)


def show_results(
    img_paths: Sequence[Path] | pd.Index,
    algo: Callable[[PIL.Image.Image], tuple[int, int, int]],
    init_i: int = 0,
) -> None:
    if isinstance(img_paths, pd.Index):
        img_paths = cast(Sequence[Path], img_paths)

    def f(i: int) -> None:
        print(f"Image {i + 1}/{len(img_paths)}: {img_paths[i].name}")
        if img_paths[i] in GROUND_TRUTHS:
            result = run_and_compare(img_paths[i], algo=algo)
            display(show_result(result, GROUND_TRUTHS[img_paths[i]]))
        else:
            result = run_algorithm(img_paths[i], algo=algo)
            display(show_result(result))

    ipywidgets.interact(
        f, i=ipywidgets.IntSlider(min=0, max=len(img_paths) - 1, value=init_i)
    )

# Cell 14
def find_circle_baseline(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    cy, cx, r = (
        pil_img.height // 2,
        pil_img.width // 2,
        min(pil_img.height, pil_img.width) // 2,
    )
    return cy, cx, r

# Cell 15
results = run_all(val_set, algo=find_circle_baseline)
results_df = pd.DataFrame(results)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

# Cell 16
success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 18
# ordering = train_set
ordering = df.index
# ordering = (df["r"] / df["w"]).sort_values(ascending=True).index
# ordering = results_df.sort_values("iou", ascending=True).index
show_results(ordering, algo=find_circle_baseline)

# Cell 20
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 21
image = PIL.Image.open(val_set[5])
x_grad, y_grad, grad_norm = obtain_gradients(image)

plt.figure(figsize=(10,10))
plt.imshow(grad_norm)
plt.show()

# Cell 22
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        input_candidate = scalar_product / np.sqrt(grad_norm_candidate + 1e-6)
        input_candidate = min(input_candidate, 50.0)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 23
def obtain_candidates(x_min, x_max, dx, y_min, y_max, dy, r_min, r_max, dr, h, w):
  """Function that return the array with candidate circles."""

  # parameters to test
  xs = np.arange(start=x_min, stop=x_max, step=dx)
  ys = np.arange(start=y_min, stop=y_max, step=dy)
  rs = np.arange(start=r_min, stop=r_max, step=dr)

  # creating meshgrid
  X, Y, R = np.meshgrid(xs, ys, rs, indexing='ij')

  # flattening the above matrices
  X = X.ravel()
  Y = Y.ravel()
  R = R.ravel()

  # Using information: distance from corners >= 1.2r

  # computing distances from corners
  d_up_right = np.sqrt((w - X) ** 2 + Y ** 2)
  d_up_left = np.sqrt(X ** 2 + Y ** 2)
  d_down_right = np.sqrt((w - X) ** 2 + (h - Y) ** 2)
  d_down_left = np.sqrt(X ** 2 + (h - Y) ** 2)

  # minimal distance from corners
  min_dist = np.minimum(np.minimum(np.minimum(d_up_right, d_up_left), d_down_right), d_down_left)

  mask = (min_dist >= 1.2 * R)

  # filtering using mask
  X = X[mask]
  Y = Y[mask]
  R = R[mask]

  # joining the matrices into one matrix
  matrix_combined = np.column_stack((Y, X, R))

  # adjusting type
  matrix_combined = matrix_combined.astype(np.float32)

  return matrix_combined

# Cell 24
def find_circle_A(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    big_step = 15
    candidates_phase_1 = obtain_candidates(x_min=0, x_max=w, dx=big_step, y_min=0, y_max=h, dy=big_step, r_min=300, r_max=max(h, w) // 2+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 25
img_path = val_set[70]
image = PIL.Image.open(img_path)
print(image.height, image.width)

result = run_and_compare(img_path, algo=find_circle_A)
gt = GROUND_TRUTHS[img_path]
show_result(result, gt=gt)

# Cell 26
results_A = run_all(val_set, algo=find_circle_A)
results_A

# Cell 27
results_df = pd.DataFrame(results_A)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

# Cell 28
# new
success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 29
# old
success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 30
def find_circle_A_2(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 15
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 31
# new
results_A_2 = run_all(val_set, algo=find_circle_A_2)

results_df = pd.DataFrame(results_A_2)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 32
# old
results_A_2 = run_all(val_set, algo=find_circle_A_2)

results_df = pd.DataFrame(results_A_2)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 33
def find_circle_A_3(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 7
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 34
# new
results_A_3 = run_all(val_set, algo=find_circle_A_3)

results_df = pd.DataFrame(results_A_3)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 35
# old
results_A_3 = run_all(val_set, algo=find_circle_A_3)

results_df = pd.DataFrame(results_A_3)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 36
def find_circle_A_4(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=10.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 37
# new
results_A_4 = run_all(val_set, algo=find_circle_A_4)

results_df = pd.DataFrame(results_A_4)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 38
# old
results_A_4 = run_all(val_set, algo=find_circle_A_4)

results_df = pd.DataFrame(results_A_4)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 39
def find_circle_A_5(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 5
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 40
# new
results_A_5 = run_all(val_set, algo=find_circle_A_5)

results_df = pd.DataFrame(results_A_5)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 41
def find_circle_A_6(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 7
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 42
# new
results_A_6 = run_all(val_set, algo=find_circle_A_6)

results_df = pd.DataFrame(results_A_6)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 43
# testing lower `big_step` value in regard to 6th pipeline
def find_circle_A_7(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 44
# new
results_A_7 = run_all(val_set, algo=find_circle_A_7)

results_df = pd.DataFrame(results_A_7)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 45
# testing higher `blur_kernel_size` in regard to 6th pipeline
def find_circle_A_8(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img, blur_kernel_size=21)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 7
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 46
# new
results_A_8 = run_all(val_set, algo=find_circle_A_8)

results_df = pd.DataFrame(results_A_8)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 47
# testing lower `blur_kernel_size` in regard to 6th pipeline
def find_circle_A_9(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img, blur_kernel_size=7)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 7
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 48
# new
results_A_9 = run_all(val_set, algo=find_circle_A_9)

results_df = pd.DataFrame(results_A_9)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 49
# testing much higher `blur_kernel_size` in regard to 6th pipeline
def find_circle_A_10(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img, blur_kernel_size=31)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 7
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 50
# new
results_A_10 = run_all(val_set, algo=find_circle_A_10)

results_df = pd.DataFrame(results_A_10)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 52
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.medianBlur(image_binary, blur_kernel_size)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 53
# testing median blur instead of gaussian blur
def find_circle_A_11(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 54
# new
results_A_11 = run_all(val_set, algo=find_circle_A_11)

results_df = pd.DataFrame(results_A_11)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 56
# testing higher `search_radius`
def find_circle_A_12(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=1.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=1.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 57
# new
results_A_12 = run_all(val_set, algo=find_circle_A_12)

results_df = pd.DataFrame(results_A_12)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 58
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        input_candidate = scalar_product / np.sqrt(grad_norm_candidate + 1e-6)
        input_candidate = min(input_candidate, 100.0)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 59
# testing new `scoring_function`
def find_circle_A_13(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 60
# new
results_A_13 = run_all(val_set, algo=find_circle_A_13)

results_df = pd.DataFrame(results_A_13)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 61
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate

        candidate_sum += cos_angle

    # updating results
    results[i] = candidate_sum

  return results

# Cell 62
# testing new `scoring_function`
def find_circle_A_14(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 63
# new
results_A_14 = run_all(val_set, algo=find_circle_A_14)

results_df = pd.DataFrame(results_A_14)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 64
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 3) * np.sqrt(grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 65
# testing new `scoring_function`
def find_circle_A_15(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 66
# THIS HAS GREAT POTENTIAL
results_A_15 = run_all(val_set, algo=find_circle_A_15)

results_df = pd.DataFrame(results_A_15)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 67
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        x = grad_norm_candidate / 20.0
        weight = 1.0 / (1.0 + np.exp(-x))
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = cos_angle * weight

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 68
# testing new `scoring_function`
def find_circle_A_16(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 69
# new
results_A_16 = run_all(val_set, algo=find_circle_A_16)

results_df = pd.DataFrame(results_A_16)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 70
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_kernel_size, blur_kernel_size))
  image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.medianBlur(image_binary, blur_kernel_size)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 71
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        input_candidate = scalar_product / np.sqrt(grad_norm_candidate + 1e-6)
        input_candidate = min(input_candidate, 50.0)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 72
# testing new `scoring_function`
def find_circle_A_17(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 73
# new
results_A_17 = run_all(val_set, algo=find_circle_A_17)

results_df = pd.DataFrame(results_A_17)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 74
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 5) * (grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 75
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 76
# testing new `scoring_function`
def find_circle_A_18(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 77
# new
results_A_18 = run_all(val_set, algo=find_circle_A_18)

results_df = pd.DataFrame(results_A_18)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 78
# ordering = train_set
ordering = df.index
# ordering = (df["r"] / df["w"]).sort_values(ascending=True).index
# ordering = results_df.sort_values("iou", ascending=True).index
show_results(ordering, algo=find_circle_A_18)

# Cell 79
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_kernel_size, blur_kernel_size))
  image_gray = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.medianBlur(image_binary, blur_kernel_size)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 80
# testing new `scoring_function`
def find_circle_A_19(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 81
# new
results_A_19 = run_all(val_set, algo=find_circle_A_19)

results_df = pd.DataFrame(results_A_19)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 82
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 5) * (grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 83
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 84
# testing new `scoring_function`
def find_circle_A_20(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 85
# new
results_A_20 = run_all(val_set, algo=find_circle_A_20)

results_df = pd.DataFrame(results_A_20)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 86
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 87
# testing new `scoring_function`
def find_circle_A_21(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 88
# new
results_A_21 = run_all(val_set, algo=find_circle_A_21)

results_df = pd.DataFrame(results_A_21)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 89
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(6,6))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 90
# testing new `scoring_function`
def find_circle_A_22(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 91
# new
results_A_22 = run_all(val_set, algo=find_circle_A_22)

results_df = pd.DataFrame(results_A_22)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 92
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(6,6))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 93
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 7) * (grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 94
# testing new `scoring_function`
def find_circle_A_23(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 95
# new
results_A_23 = run_all(val_set, algo=find_circle_A_23)

results_df = pd.DataFrame(results_A_23)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 96
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        cos_score = cos_angle ** 5
        classic_score = min(scalar_product / np.sqrt(grad_norm_candidate + 1e-6), 50.0)
        input_candidate = cos_score + classic_score

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 97
# testing new `scoring_function`
def find_circle_A_24(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 98
# new
results_A_24 = run_all(val_set, algo=find_circle_A_24)

results_df = pd.DataFrame(results_A_24)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 100
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 5) * (grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 101
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 102
# testing new `scoring_function`
def find_circle_A_25(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=256, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 103
# new
results_A_25 = run_all(val_set, algo=find_circle_A_25)

results_df = pd.DataFrame(results_A_25)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 104
# testing new `scoring_function`
def find_circle_A_26(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=512, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 105
# new
results_A_26 = run_all(val_set, algo=find_circle_A_26)

results_df = pd.DataFrame(results_A_26)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 107
# this is an attempt to make the pipeline faster, while being more precise -> introducing fast_check
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0, n_fast_points=32, threshold_2=2.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points
  d_angle_fast = 2 * np.pi / n_fast_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    fast_candidate_sum = 0.0

    for k in range(n_fast_points):
      # updating angle
      angle = d_angle_fast * k

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 5) * (grad_norm_candidate)

        fast_candidate_sum += input_candidate

    # skipping this candidate if it didn't perform during fast check
    if fast_candidate_sum < threshold_2:
      results[i] = 0
      continue

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 5) * (grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum + fast_candidate_sum

  return results

# Cell 108
# testing new `scoring_function`
def find_circle_A_27(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step, dr=big_step, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=512, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 109
# new
results_A_27 = run_all(val_set, algo=find_circle_A_27)

results_df = pd.DataFrame(results_A_27)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 110
@numba.njit
def scoring_function(candidates, x_grad, y_grad, grad_norm, n_points, threshold=1.0):
  """Function designed to calculate scores for candidate circles based on the strength of the gradient."""
  h, w = x_grad.shape
  n_candidates = len(candidates)

  # creating placeholder for results
  results = np.zeros(n_candidates)

  # angle step
  d_angle = 2 * np.pi / n_points

  for i in range(candidates.shape[0]):
    # obtaining details about candidate
    cy = candidates[i, 0]
    cx = candidates[i, 1]
    r = candidates[i, 2]

    candidate_sum = 0.0

    for j in range(n_points):
      # updating angle
      angle = d_angle * j

      # calculating position of point
      x = cx + r * np.cos(angle)
      y = cy + r * np.sin(angle)

      # rounding to get the nearest pixel
      x = int(np.round(x))
      y = int(np.round(y))

      # checking whether we didn't fall outside the image
      if x < 0 or x >= w or y < 0 or y >= h:
        continue

      # getting gradients from passed `x_grad` and `y_grad`
      x_grad_candidate = x_grad[y, x]
      y_grad_candidate = y_grad[y, x]
      grad_norm_candidate = grad_norm[y, x]

      # to filter out the noise we only consider cases when gradient is strong enough
      if grad_norm_candidate < threshold:
        continue

      # vector that should be pointing at the center
      x_vector = -np.cos(angle)
      y_vector = -np.sin(angle)

      # computing scalar product
      scalar_product = x_vector * x_grad_candidate + y_vector * y_grad_candidate

      # checking if gradient vector is pointed in correct direction
      if scalar_product > 0:
        # mitigating the impact of huge gradients by dividing by the square root of gradient norm
        # calculating the input of this point for this candidate
        cos_angle = scalar_product / grad_norm_candidate
        input_candidate = (cos_angle ** 5) * (grad_norm_candidate)

        candidate_sum += input_candidate

    # updating results
    results[i] = candidate_sum

  return results

# Cell 111
def obtain_gradients(image, blur_kernel_size=15):
  """Helper function that prepares gradients for `find_circle_A` for further work."""

  image_np = np.array(image)

  # converting to grayscale
  image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

  # NEW UPDATE
  clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
  image_gray = clahe.apply(image_gray)

  # converting to binary image: non-black pixel becomes white
  image_binary = image_gray
  #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

  # applying smoothing to reduce the impact of noise/structure of the eye itself
  image_smoothed = cv2.GaussianBlur(image_binary, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

  # computing gradients using `cv2.Scharr`
  x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
  y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

  # TODO information that in some way will be used to mitigate the influence of gradient
  grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

  return (x_grad, y_grad, grad_norm)

# Cell 112
# testing new `scoring_function`
def find_circle_A_28(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)

    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img, blur_kernel_size=21)

    # Phase 1
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    big_step = 6
    big_step_r = 4
    candidates_phase_1 = obtain_candidates(x_min=x_min, x_max=x_max, dx=big_step, y_min=y_min, y_max=y_max, dy=big_step, r_min=r_min, r_max=r_max+big_step_r, dr=big_step_r, h=h, w=w)
    results_phase_1 = scoring_function(candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=64, threshold=5.0)
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # Phase 2
    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, h=h, w=w)
    results_phase_2 = scoring_function(candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, grad_norm=grad_norm, n_points=128, threshold=5.0)
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    # adjusting type
    cx = int(cx)
    cy = int(cy)
    r = int(r)

    return cy, cx, r

# Cell 113
# new
results_A_28 = run_all(val_set, algo=find_circle_A_28)

results_df = pd.DataFrame(results_A_28)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 115
def find_circle_B(pil_img: PIL.Image.Image, param1=100, param2=40) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    # converting to grayscale
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # applying CLAHE
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))
    image_gray = clahe.apply(image_gray)

    # converting to binary image: non-black pixel becomes white
    image_binary = image_gray
    #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

    # applying smoothing to reduce the impact of noise/structure of the eye itself
    image_smoothed = cv2.GaussianBlur(image_binary, ksize=(21, 21), sigmaX=0)

    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))

    circles = cv2.HoughCircles(image_smoothed, cv2.HOUGH_GRADIENT, minDist=min(h, w), dp=1, param1=param1, param2=param2, minRadius=r_min, maxRadius=r_max)

    if circles is not None:
      circle = circles[0]
      cx, cy, r =  circle[0]

    else:
      cy, cx, r = h // 2, w // 2, min(h, w)// 2

    cx = int(np.round(cx))
    cy = int(np.round(cy))
    r = int(np.round(r))
    return cy, cx, r

# Cell 116
def find_circle_B(pil_img: PIL.Image.Image, param1: int, param2: int) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    # converting to grayscale
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # applying CLAHE
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4,4))
    image_gray = clahe.apply(image_gray)

    # converting to binary image: non-black pixel becomes white
    image_binary = image_gray
    #_, image_binary = cv2.threshold(image_gray, thresh=10, maxval=255, type=cv2.THRESH_BINARY)

    # applying smoothing to reduce the impact of noise/structure of the eye itself
    image_smoothed = cv2.GaussianBlur(image_binary, ksize=(21, 21), sigmaX=0)

    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))

    circles = cv2.HoughCircles(image_smoothed, cv2.HOUGH_GRADIENT, minDist=min(h, w), dp=1, param1=param1, param2=param2, minRadius=r_min, maxRadius=r_max)

    if circles is not None:
      circle = circles[0]
      cx, cy, r =  circle[0]

    else:
      cy, cx, r = h // 2, w // 2, min(h, w)// 2

    cx = int(np.round(cx))
    cy = int(np.round(cy))
    r = int(np.round(r))
    return cy, cx, r

# Cell 117
# new
results_B = run_all(val_set, algo=find_circle_B)

results_df = pd.DataFrame(results_B)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 118
grid_param1 = [60, 80, 100, 120, 140, 160]
grid_param2 = [25, 30, 35, 40, 45, 55]
for p1, p2 in itertools.product(grid_param1, grid_param2):
  print(f"================{p1}, {p2}==========================")
  algorithm = partial(find_circle_B, param1=p1, param2=p2)
  results_B = run_all(val_set, algo=algorithm)

  results_df = pd.DataFrame(results_B)
  results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
  results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
  results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
  results_df

  success_rate = (results_df["iou"] >= 0.85).mean()
  accuracy = results_df["iou"].clip(0.85, 0.98).mean()
  print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")


# Cell 120
def create_ground_truth(train_set, algorithm):

  # list for storing ground truths
  ground_truths = []

  # creating ground thruth for every element of training dataset
  for path in tqdm(train_set):
    image = PIL.Image.open(path)
    cy, cx, r = algorithm(image)
    ground_truths.append({"image_path": str(path), "cy": cy, "cx": cx, "r": r})

  # saving to `csv` file
  df = pd.DataFrame(ground_truths)
  df.to_csv("ground_truths.csv", index=False)

  return df

# Cell 121
ground_truths_train_df = create_ground_truth(train_set, find_circle_A_28)

# Cell 122
size = 128

class Eye_Dataset(Dataset):
  def __init__(self, csv_file):
    self.labels = pd.read_csv(csv_file)
    self.transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, id):
    row = self.labels.iloc[id]
    image_path = row["image_path"]
    image = PIL.Image.open(image_path).convert("RGB")
    w, h = image.size
    image_transformed = self.transforms(image)

    # normalizing labels
    cx_norm = float(row["cx"]) / w
    cy_norm = float(row["cy"]) / h
    r_norm = float(row["r"]) / max(h, w)

    labels_tensor = torch.tensor([cy_norm, cx_norm, r_norm], dtype=torch.float32)

    return image_transformed, labels_tensor


# Cell 123
dataset = Eye_Dataset(csv_file="ground_truths.csv")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Cell 124
# cell with constructing our model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(num_features, 3)

# Cell 125
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Cell 127
import matplotlib.pyplot as plt

def train_and_validate(model, train_loader, val_loader, n_epochs):
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Listy do przechowywania historii
    history = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        # --- TRENING ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_mse(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- WALIDACJA ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                loss = loss_mse(pred, labels)
                val_loss += loss.item()

        # Obliczanie średnich
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        history['train'].append(avg_train)
        history['val'].append(avg_val)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

    return history

# --- RYSOWANIE WYKRESU ---
def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.title('Krzywe uczenia (Learning Curves)')
    plt.xlabel('Epoka')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cell 128
class ValDataset(Dataset):
    def __init__(self, val_paths, ground_truths, transform):
        self.paths = val_paths
        self.gt = ground_truths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]

        # 1. Wczytanie obrazu
        image = PIL.Image.open(img_path).convert("RGB")
        w, h = image.size

        # 2. Transformacja obrazu (ta sama co w train!)
        image_transformed = self.transform(image)

        # 3. Pobranie etykiety z Twojego słownika GROUND_TRUTHS
        data = self.gt[img_path]

        # 4. Normalizacja etykiet (musi być identyczna jak w TrainDataset)
        cy_norm = float(data["cy"]) / h
        cx_norm = float(data["cx"]) / w
        r_norm = float(data["r"]) / max(h, w)

        labels_tensor = torch.tensor([cy_norm, cx_norm, r_norm], dtype=torch.float32)

        return image_transformed, labels_tensor

# Cell 129
# 1. Tworzymy Dataset
val_dataset = ValDataset(val_set, GROUND_TRUTHS, transform=transforms_B)

# 2. Tworzymy DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,      # WAŻNE: W walidacji nie mieszamy!
    num_workers=2,
    pin_memory=True
)

# Cell 130
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
history = train_and_validate(model=model, train_loader=train_dataloader, val_loader=val_loader, n_epochs=100)


# Cell 131
plot_history(history)

# Cell 132
SEKCJA STOP

# Cell 133
def train_network(model, dataloader, n_epochs):
  loss_mse = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  model.train()
  for epoch in range(n_epochs):
    loss_sum = 0.0
    for images, labels in dataloader:
      images = images.to(device)
      labels = labels.to(device)

      predictions = model(images)

      loss = loss_mse(predictions, labels)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      loss_sum += loss.item()

    loss_avg = loss_sum / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}: Average Loss = {loss_avg}")

  return model

# Cell 134
train_network(model=model, dataloader=dataloader, n_epochs=30)

# Cell 135
def evaluate_model(model, dataloader):
  model.eval()
  results = []

  with torch.no_grad():
    for images, labels in dataloader:
      predictions = model(images)

      for i in range(len(predictions)):
        prediction = prediction[i].cpu().numpy()

        result.append(prediction)

  return results

# Cell 136
transforms_B = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])

# Cell 137
device_B = torch.device("cpu")
model.to(device_B)

# Cell 138
def find_circle_B(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    #cy, cx, r = find_circle_baseline(pil_img)
    device_B = torch.device("cpu")
    #model.to(device_B)
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    model.eval()
    image_tensor = transforms_B(pil_img).unsqueeze(0).to(device_B)
    with torch.no_grad():
      output = model(image_tensor)
      prediction = output.squeeze(0).cpu().numpy()

    cy = np.clip(prediction[0], 0, 1) * h
    cx = np.clip(prediction[1], 0, 1) * w
    r = np.clip(prediction[2], 0, max(w,h)) * max(h, w)

    cx = int(np.round(cx))
    cy = int(np.round(cy))
    r = int(np.round(r))
    return cy, cx, r

# Cell 139
# new
results_B = run_all(val_set, algo=find_circle_B)

results_df = pd.DataFrame(results_B)
results_df["difficulty"] = results_df["path"].map(lambda p: p.split("/")[1])
results_df["index"] = results_df["path"].apply(lambda p: IMG_DIR / p)
results_df = results_df.drop_duplicates(subset="index", keep="last").set_index("index")
results_df

success_rate = (results_df["iou"] >= 0.85).mean()
accuracy = results_df["iou"].clip(0.85, 0.98).mean()
print(f"Success Rate: {success_rate:.1%}, Accuracy: {accuracy:.2%}")

# Cell 140
# ordering = train_set
ordering = df.index
# ordering = (df["r"] / df["w"]).sort_values(ascending=True).index
# ordering = results_df.sort_values("iou", ascending=True).index
show_results(ordering, algo=find_circle_B)