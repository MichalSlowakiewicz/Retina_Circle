import cv2
import numpy as np
import PIL.Image
from src.algorithms.gradient import obtain_candidates, scoring_function

def obtain_gradients_experimental(image: PIL.Image.Image, blur_kernel_size: int = 15, use_median_blur: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to obtain gradients, allowing experimentation with different blurs."""
    image_np = np.array(image)
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    image_gray = clahe.apply(image_gray)
    
    if use_median_blur:
        image_smoothed = cv2.medianBlur(image_gray, blur_kernel_size)
    else:
        image_smoothed = cv2.GaussianBlur(image_gray, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=0)
        
    x_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=1, dy=0)
    y_grad = cv2.Scharr(image_smoothed, ddepth=cv2.CV_32F, dx=0, dy=1)
    
    grad_norm = np.sqrt(x_grad ** 2 + y_grad ** 2)
    return x_grad, y_grad, grad_norm


def find_circle_A_baseline(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Early version (A_1 to A_7 era). 
    Uses GaussianBlur(15), big_step=7 (or 15 initially), and a basic scalar product for scoring.
    """
    h, w = pil_img.height, pil_img.width
    x_grad, y_grad, grad_norm = obtain_gradients_experimental(pil_img, blur_kernel_size=15)

    y_min, y_max, x_min, x_max = int(0.3*h), int(0.7*h), int(0.3*w), int(0.7*w)
    r_min, r_max = int(0.25*max(h,w)), int(0.75*min(h,w))

    big_step = 7
    candidates_phase_1 = obtain_candidates(x_min, x_max, big_step, y_min, y_max, big_step, r_min, r_max+big_step, big_step, h, w)
    
    # Original scoring logic was slightly different, but we use our compiled one with a small threshold
    results_phase_1 = scoring_function(candidates_phase_1, x_grad, y_grad, grad_norm, n_points=64, threshold=1.0)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[np.argmax(results_phase_1)]

    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(
        x_phase_1-search_radius, x_phase_1+search_radius+1, small_step, 
        y_phase_1-search_radius, y_phase_1+search_radius+1, small_step, 
        r_phase_1-search_radius, r_phase_1+search_radius+1, small_step, h, w
    )
    results_phase_2 = scoring_function(candidates_phase_2, x_grad, y_grad, grad_norm, n_points=128, threshold=1.0)
    cy, cx, r = candidates_phase_2[np.argmax(results_phase_2)]

    return int(cy), int(cx), int(r)


def find_circle_A_11_median(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Iteration A_11: Testing Median Blur instead of Gaussian Blur to handle eye reflections.
    """
    h, w = pil_img.height, pil_img.width
    x_grad, y_grad, grad_norm = obtain_gradients_experimental(pil_img, blur_kernel_size=15, use_median_blur=True)

    y_min, y_max, x_min, x_max = int(0.3*h), int(0.7*h), int(0.3*w), int(0.7*w)
    r_min, r_max = int(0.25*max(h,w)), int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min, x_max, big_step, y_min, y_max, big_step, r_min, r_max+big_step, big_step, h, w)
    results_phase_1 = scoring_function(candidates_phase_1, x_grad, y_grad, grad_norm, n_points=64, threshold=1.0)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[np.argmax(results_phase_1)]

    small_step = 1
    search_radius = big_step
    candidates_phase_2 = obtain_candidates(
        x_phase_1-search_radius, x_phase_1+search_radius+1, small_step, 
        y_phase_1-search_radius, y_phase_1+search_radius+1, small_step, 
        r_phase_1-search_radius, r_phase_1+search_radius+1, small_step, h, w
    )
    results_phase_2 = scoring_function(candidates_phase_2, x_grad, y_grad, grad_norm, n_points=128, threshold=1.0)
    cy, cx, r = candidates_phase_2[np.argmax(results_phase_2)]

    return int(cy), int(cx), int(r)


def find_circle_A_15_radius(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Iteration A_15: Increased search_radius to 15 (from 6) and threshold to 5.0 to filter out weak gradients.
    Also used cos_angle ** 3 to penalize misalignment.
    """
    h, w = pil_img.height, pil_img.width
    x_grad, y_grad, grad_norm = obtain_gradients_experimental(pil_img, blur_kernel_size=15)

    y_min, y_max, x_min, x_max = int(0.3*h), int(0.7*h), int(0.3*w), int(0.7*w)
    r_min, r_max = int(0.25*max(h,w)), int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min, x_max, big_step, y_min, y_max, big_step, r_min, r_max+big_step, big_step, h, w)
    results_phase_1 = scoring_function(candidates_phase_1, x_grad, y_grad, grad_norm, n_points=64, threshold=5.0)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[np.argmax(results_phase_1)]

    small_step = 1
    search_radius = 15  # <--- Crucial change
    candidates_phase_2 = obtain_candidates(
        x_phase_1-search_radius, x_phase_1+search_radius+1, small_step, 
        y_phase_1-search_radius, y_phase_1+search_radius+1, small_step, 
        r_phase_1-search_radius, r_phase_1+search_radius+1, small_step, h, w
    )
    results_phase_2 = scoring_function(candidates_phase_2, x_grad, y_grad, grad_norm, n_points=128, threshold=5.0)
    cy, cx, r = candidates_phase_2[np.argmax(results_phase_2)]

    return int(cy), int(cx), int(r)


def find_circle_A_27_dense(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Iteration A_27: High precision testing.
    Uses n_points=128 in Phase 1, and n_points=512 in Phase 2.
    Very dense, but computationally expensive.
    """
    h, w = pil_img.height, pil_img.width
    x_grad, y_grad, grad_norm = obtain_gradients_experimental(pil_img, blur_kernel_size=15)

    y_min, y_max, x_min, x_max = int(0.3*h), int(0.7*h), int(0.3*w), int(0.7*w)
    r_min, r_max = int(0.25*max(h,w)), int(0.75*min(h,w))

    big_step = 6
    candidates_phase_1 = obtain_candidates(x_min, x_max, big_step, y_min, y_max, big_step, r_min, r_max+big_step, big_step, h, w)
    results_phase_1 = scoring_function(candidates_phase_1, x_grad, y_grad, grad_norm, n_points=128, threshold=5.0)  # Dense
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[np.argmax(results_phase_1)]

    small_step = 1
    search_radius = 15
    candidates_phase_2 = obtain_candidates(
        x_phase_1-search_radius, x_phase_1+search_radius+1, small_step, 
        y_phase_1-search_radius, y_phase_1+search_radius+1, small_step, 
        r_phase_1-search_radius, r_phase_1+search_radius+1, small_step, h, w
    )
    results_phase_2 = scoring_function(candidates_phase_2, x_grad, y_grad, grad_norm, n_points=512, threshold=5.0)  # Very Dense
    cy, cx, r = candidates_phase_2[np.argmax(results_phase_2)]

    return int(cy), int(cx), int(r)
