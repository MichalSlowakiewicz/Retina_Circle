import cv2
import numba
import numpy as np
import PIL.Image

def obtain_gradients(image: PIL.Image.Image, blur_kernel_size: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the image gradients for further processing (voting-based algorithm).
    CLAHE and GaussianBlur are applied to minimize noise and the impact of the eye's natural structure.
    """
    image_np = np.array(image)

    # Convert to grayscale
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)

    # Gaussian blur to reduce noise
    image_smoothed = cv2.GaussianBlur(image_gray, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)

    # Computing gradients using the Scharr filter (more accurate than Sobel)
    x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
    y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)

    # Gradient norm
    grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)

    return x_grad, y_grad, grad_norm

def obtain_candidates(
    x_min: int, x_max: int, dx: int,
    y_min: int, y_max: int, dy: int,
    r_min: int, r_max: int, dr: int,
    h: int, w: int
) -> np.ndarray:
    """Generates a matrix containing circle candidate proposals (y, x, r)."""
    xs = np.arange(start=x_min, stop=x_max, step=dx)
    ys = np.arange(start=y_min, stop=y_max, step=dy)
    rs = np.arange(start=r_min, stop=r_max, step=dr)

    X, Y, R = np.meshgrid(xs, ys, rs, indexing='ij')

    X = X.ravel()
    Y = Y.ravel()
    R = R.ravel()

    # Apriori information: the distance of the circle from the image corners should be >= 1.2 * r
    d_up_right = np.sqrt((w - X) ** 2 + Y ** 2)
    d_up_left = np.sqrt(X ** 2 + Y ** 2)
    d_down_right = np.sqrt((w - X) ** 2 + (h - Y) ** 2)
    d_down_left = np.sqrt(X ** 2 + (h - Y) ** 2)

    min_dist = np.minimum(np.minimum(np.minimum(d_up_right, d_up_left), d_down_right), d_down_left)
    mask = (min_dist >= 1.2 * R)

    X = X[mask]
    Y = Y[mask]
    R = R[mask]

    matrix_combined = np.column_stack((Y, X, R))
    return matrix_combined.astype(np.float32)

@numba.njit(fastmath=True)
def scoring_function(
    candidates: np.ndarray, x_grad: np.ndarray, y_grad: np.ndarray, 
    grad_norm: np.ndarray, n_points: int, threshold: float = 1.0
) -> np.ndarray:
    """
    Scoring function (optimized via Numba) that assigns a score to circle candidates
    based on the strength of gradients along their perimeter pointing inwards.
    """
    h, w = x_grad.shape
    n_candidates = len(candidates)
    results = np.zeros(n_candidates)
    d_angle = 2 * np.pi / n_points

    for i in range(candidates.shape[0]):
        cy = candidates[i, 0]
        cx = candidates[i, 1]
        r = candidates[i, 2]

        candidate_sum = 0.0

        for j in range(n_points):
            angle = d_angle * j
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            x_idx = int(np.round(x))
            y_idx = int(np.round(y))

            if x_idx < 0 or x_idx >= w or y_idx < 0 or y_idx >= h:
                continue

            x_grad_c = x_grad[y_idx, x_idx]
            y_grad_c = y_grad[y_idx, x_idx]
            grad_norm_c = grad_norm[y_idx, x_idx]

            # Filter out noise using a threshold on the gradient norm
            if grad_norm_c < threshold:
                continue

            # Vector pointing from the perimeter towards the supposed center of the circle
            x_vector = -np.cos(angle)
            y_vector = -np.sin(angle)

            scalar_product = x_vector * x_grad_c + y_vector * y_grad_c

            if scalar_product > 0:
                # Mitigate the impact of huge gradients by dividing by the norm,
                # and penalize vectors that don't align perfectly (using cos_angle^5)
                cos_angle = scalar_product / grad_norm_c
                input_candidate = (cos_angle ** 5) * grad_norm_c
                candidate_sum += input_candidate

        results[i] = candidate_sum

    return results

def find_circle_gradient(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    The final, best gradient-based detection algorithm (originating from iteration A_28).
    Split into two phases: Phase 1 (coarse search) and Phase 2 (fine search).
    """
    h = pil_img.height
    w = pil_img.width

    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img, blur_kernel_size=21)

    # --- Phase 1: Coarse search (large step, wide search space) ---
    y_min, y_max = int(0.3 * h), int(0.7 * h)
    x_min, x_max = int(0.3 * w), int(0.7 * w)
    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))

    big_step = 6
    big_step_r = 4
    
    candidates_phase_1 = obtain_candidates(
        x_min=x_min, x_max=x_max, dx=big_step, 
        y_min=y_min, y_max=y_max, dy=big_step, 
        r_min=r_min, r_max=r_max + big_step_r, dr=big_step_r, 
        h=h, w=w
    )
    results_phase_1 = scoring_function(
        candidates_phase_1, x_grad, y_grad, grad_norm, n_points=64, threshold=5.0
    )
    
    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # --- Phase 2: Fine search around the best candidate from Phase 1 ---
    small_step = 1
    search_radius = 15
    
    candidates_phase_2 = obtain_candidates(
        x_min=x_phase_1 - search_radius, x_max=x_phase_1 + search_radius + 1, dx=small_step, 
        y_min=y_phase_1 - search_radius, y_max=y_phase_1 + search_radius + 1, dy=small_step, 
        r_min=r_phase_1 - search_radius, r_max=r_phase_1 + search_radius + 1, dr=small_step, 
        h=h, w=w
    )
    results_phase_2 = scoring_function(
        candidates_phase_2, x_grad, y_grad, grad_norm, n_points=128, threshold=5.0
    )
    
    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    return int(cy), int(cx), int(r)
