import PIL.Image
import numpy as np

# We reuse the highly optimized helper functions from the standard gradient module
from src.algorithms.gradient import obtain_gradients, obtain_candidates, scoring_function

def find_circle_fast_gradient(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    A much faster variant of the Gradient-Based detection algorithm.
    It trades a tiny fraction of density in the search grid for a massive speedup,
    while still being remarkably robust at handling extreme edge cases in fundus photographs.
    """
    h = pil_img.height
    w = pil_img.width

    # calculating gradients
    x_grad, y_grad, grad_norm = obtain_gradients(image=pil_img, blur_kernel_size=21)

    # --- Phase 1: Coarse search for potential cx, cy, r ---
    y_min = int(0.3*h)
    y_max = int(0.7*h)
    x_min = int(0.3*w)
    x_max = int(0.7*w)
    r_min = int(0.25*max(h,w))
    r_max = int(0.75*min(h,w))

    # Using larger steps (8 and 5) compared to the standard approach (6 and 4)
    # This exponentially reduces the number of initial candidates to evaluate.
    big_step = 8
    big_step_r = 5

    candidates_phase_1 = obtain_candidates(
        x_min=x_min, x_max=x_max, dx=big_step, 
        y_min=y_min, y_max=y_max, dy=big_step, 
        r_min=r_min, r_max=r_max+big_step_r, dr=big_step_r, 
        h=h, w=w
    )
    
    # We also evaluate only 32 points along the perimeter instead of 64
    results_phase_1 = scoring_function(
        candidates=candidates_phase_1, x_grad=x_grad, y_grad=y_grad, 
        grad_norm=grad_norm, n_points=32, threshold=5.0
    )

    id_max = np.argmax(results_phase_1)
    y_phase_1, x_phase_1, r_phase_1 = candidates_phase_1[id_max]

    # --- Phase 2: Precise search for cx, cy, r ---
    small_step = 1
    search_radius = 15

    candidates_phase_2 = obtain_candidates(
        x_min=x_phase_1-search_radius, x_max=x_phase_1+search_radius+1, dx=small_step, 
        y_min=y_phase_1-search_radius, y_max=y_phase_1+search_radius+1, dy=small_step, 
        r_min=r_phase_1-search_radius, r_max=r_phase_1+search_radius+1, dr=small_step, 
        h=h, w=w
    )
    
    # In the final phase, we use 128 points to ensure perfect precision.
    results_phase_2 = scoring_function(
        candidates=candidates_phase_2, x_grad=x_grad, y_grad=y_grad, 
        grad_norm=grad_norm, n_points=128, threshold=5.0
    )

    id_max = np.argmax(results_phase_2)
    cy, cx, r = candidates_phase_2[id_max]

    return int(cy), int(cx), int(r)
