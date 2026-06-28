# Algorithms Documentation 🧠

This directory contains all the algorithms developed for the Eye Circle Detection task. The journey spanned from baseline classical methods to heavily optimized custom implementations, and finally to deep learning.

Here is a breakdown of what each file does:

### 1. `hough.py` (Final Algorithm B)
This file contains our highly optimized baseline, often referred to as **Algorithm B**. 
* **Methodology:** It utilizes OpenCV's `cv2.HoughCircles` (`HOUGH_GRADIENT`) combined with a very strong median blur (kernel size 21) and contrast enhancement (CLAHE).
* **Performance:** With a meticulously tuned accumulator threshold (`param2=3`), this algorithm achieves an impressive **~98% Success Rate**. It is incredibly robust against noisy fundus photographs.

### 2. `gradient.py` (Final Algorithm A)
This file houses our custom-built, Numba-optimized gradient voting algorithm, referred to as **Algorithm A**.
* **Methodology:** It computes the image gradients using Scharr filters and scores thousands of potential circle candidates based on the alignment of the gradient vectors along the circle's perimeter. It runs a two-phase search (coarse and refined) to quickly locate the best fit.
* **Performance:** Achieves a **~96% Success Rate**, acting as an excellent deterministic fallback that doesn't rely on OpenCV's internal Canny edge detection.

### 3. `fast_gradient.py` (Algorithm A - Speed Optimized)
A heavily optimized variation of Algorithm A, focusing on pure speed and extreme edge cases.
* **Methodology:** Reuses the core Numba helpers but dramatically reduces the number of candidate evaluations by expanding the search step (`big_step=8`) and lowering the number of evaluated points on the perimeter (`n_points=32`).
* **Performance:** Experiences a very negligible drop in IoU but executes exponentially faster, cleanly fitting within the strict execution time limits.

### 4. `neural_network.py` (Deep Learning Approach)
* **Methodology:** Contains the architecture and inference code for an `EfficientNet-B0` model. The network frames the circle detection as a regression problem, predicting normalized values for `(cy, cx, r)`.
* **Purpose:** Built to see if the complex geometric logic from `gradient.py` could be distilled into a fast, fixed-time neural network forward pass.

### 4. `experiments.py` (The Archive)
A treasure trove of alternative, experimental approaches that were developed and tested during the project. 
* **Methodology:** Contains implementations using Algebraic Least Squares fitting (`find_circle_fundus_smart`), Geometric RANSAC edge-fitting (`find_circle_RANSAC`), and pure Binary Masking with `HOUGH_GRADIENT_ALT`. It also includes the `AlgoB_Runner` used for grid-searching parameters.
* **Purpose:** Kept as a clean archive so the valuable research and alternative paths explored are preserved without cluttering the main production files.

### 5. `gradient_experiments.py`
* **Methodology:** A historical record of the iterative development of the gradient algorithm (from `find_circle_A_baseline` through various iterations like `11_median`, `15_radius`, to `27_dense`).
* **Purpose:** Documents the hyperparameter tuning and logical steps taken to reach the final Algorithm A implementation.
