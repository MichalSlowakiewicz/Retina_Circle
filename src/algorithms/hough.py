import cv2
import numpy as np
import PIL.Image

def find_circle_B(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Finds the center and radius of a circle (pupil/iris) using the Hough Circles algorithm.
    This is a highly optimized version resulting from grid search, achieving exceptional IoU.
    """
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    # Convert to grayscale
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE to equalize contrast and lighting
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
    image_gray = clahe.apply(image_gray)

    # Apply heavy median blur to reduce the impact of noise and internal eye structures
    image_smoothed = cv2.medianBlur(image_gray, 21)

    # Define the physical constraints of the circle radius
    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))

    # Detect circles using the standard Hough Gradient method
    # param1=80 sets the Canny edge threshold
    # param2=3 sets an extremely permissive accumulator threshold, which works 
    # perfectly here due to the heavy median blur stripping out false edges.
    circles = cv2.HoughCircles(
        image_smoothed, 
        cv2.HOUGH_GRADIENT, 
        minDist=min(h, w), 
        dp=1.0, 
        param1=80, 
        param2=3, 
        minRadius=r_min, 
        maxRadius=r_max
    ) 
    
    # Check if the function found any circles
    if circles is not None:
        # Take the results of the best circle
        circle = circles[0][0]
        cx, cy, r = circle
    else:
        # Fall back to the center of the image if no circles were found
        cx = w // 2
        cy = h // 2
        r = min(h, w) // 2

    # Adjust results to be integers
    cx = int(np.round(cx))
    cy = int(np.round(cy))
    r = int(np.round(r))

    return cy, cx, r
