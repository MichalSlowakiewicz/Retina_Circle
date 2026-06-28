import cv2
import numpy as np
import PIL.Image
import numba
import scipy.linalg

# ==========================================
# 1. Algebraic Least Squares Fit
# ==========================================
def fit_circle_least_squares(x, y):
    """
    Algebraic circle fitting to a cloud of points (Least Squares Method).
    Resilient to single outliers and protrusions.
    """
    # Build a system of equations: 2x*xc + 2y*yc + C = x^2 + y^2
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x**2 + y**2

    # Solve the system of equations (Returns [xc, yc, C])
    w, _, _, _ = scipy.linalg.lstsq(A, b)

    cx, cy, C = w[0], w[1], w[2]
    # Calculate radius from parameter C
    r = np.sqrt(max(0.0, C + cx**2 + cy**2))

    return int(np.round(cx)), int(np.round(cy)), int(np.round(r))

def find_circle_fundus_smart(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Experimental approach that thresholds the image to find the fundus contour, 
    then uses least squares to algebraically fit a circle to the contour points.
    """
    image_np = np.array(pil_img.convert("L"))
    h, w = image_np.shape

    # Domain constraints
    cy_min, cy_max = int(0.3 * h), int(0.7 * h)
    cx_min, cx_max = int(0.3 * w), int(0.7 * w)
    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))

    # 1. Binarization (cut off the background)
    _, thresh = cv2.threshold(image_np, 15, 255, cv2.THRESH_BINARY)

    # Morphological cleanup of the mask
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 2. Find edges
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract all X and Y coordinates from the retina edge
        points = largest_contour.reshape(-1, 2)
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 3. Fire up the math - Least Squares Fit!
        cx, cy, r = fit_circle_least_squares(x_coords, y_coords)
    else:
        cx, cy, r = w // 2, h // 2, int((r_min + r_max) / 2)

    # Hard clipping to task constraints
    cy = int(np.clip(cy, cy_min, cy_max))
    cx = int(np.clip(cx, cx_min, cx_max))
    r = int(np.clip(r, r_min, r_max))

    return cy, cx, r


# ==========================================
# 2. RANSAC Geometric Fit
# ==========================================
@numba.njit
def fit_circle_3pts(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    det = x1*(y2 - y3) - y1*(x2 - x3) + (x2*y3 - x3*y2)
    if abs(det) < 1e-6:
        return 0.0, 0.0, -1.0
    v1, v2, v3 = -(x1*x1 + y1*y1), -(x2*x2 + y2*y2), -(x3*x3 + y3*y3)
    detA = v1*(y2 - y3) - y1*(v2 - v3) + (v2*y3 - v3*y2)
    detB = x1*(v2 - v3) - v1*(x2 - x3) + (x2*v3 - x3*v2)
    detC = x1*(y2*v3 - y3*v2) - y1*(x2*v3 - x3*v2) + v1*(x2*y3 - x3*y2)
    A, B, C = detA / det, detB / det, detC / det
    cx, cy = -A / 2.0, -B / 2.0
    r = np.sqrt(max(0.0, cx*cx + cy*cy - C))
    return cy, cx, r

@numba.njit
def evaluate_circle(pts, cy, cx, r, margin):
    inliers = 0
    for i in range(len(pts)):
        y, x = pts[i, 0], pts[i, 1]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        if abs(dist - r) < margin:
            inliers += 1
    return inliers

@numba.njit
def ransac_loop(pts_array, max_iters, margin):
    n_pts = len(pts_array)
    best_inliers = -1
    best_circle = (0.0, 0.0, -1.0)
    for i in range(max_iters):
        i1 = np.random.randint(0, n_pts)
        i2 = np.random.randint(0, n_pts)
        while i2 == i1: i2 = np.random.randint(0, n_pts)
        i3 = np.random.randint(0, n_pts)
        while i3 == i1 or i3 == i2: i3 = np.random.randint(0, n_pts)
        
        p1, p2, p3 = pts_array[i1], pts_array[i2], pts_array[i3]
        cy, cx, r = fit_circle_3pts((p1[1], p1[0]), (p2[1], p2[0]), (p3[1], p3[0]))
        if r <= 0: continue
        inliers = evaluate_circle(pts_array, cy, cx, r, margin)
        if inliers > best_inliers:
            best_inliers = inliers
            best_circle = (cy, cx, r)
    return best_circle, best_inliers

def find_circle_RANSAC(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Experimental approach using RANSAC to fit a circle to edge contours.
    """
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]
    
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
    blur_k = min(31, min(h, w) // 2 | 1)
    image_smoothed = cv2.medianBlur(image_gray, blur_k)
    _, binary = cv2.threshold(image_smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best_cy, best_cx, best_r = h // 2, w // 2, min(h, w) // 2
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pts = contours[0].reshape(-1, 2)
        pts_yx = np.column_stack((pts[:, 1], pts[:, 0]))
        
        if len(pts_yx) >= 3:
            best_circle, inliers = ransac_loop(np.array(pts_yx, dtype=np.float64), 3000, 5.0)
            if inliers >= 0:
                best_cy, best_cx, best_r = best_circle

    return int(np.round(best_cy)), int(np.round(best_cx)), int(np.round(best_r))


# ==========================================
# 3. Hough Gradient ALT
# ==========================================
def find_circle_Hough_ALT(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Experimental approach using the highly precise HOUGH_GRADIENT_ALT method.
    """
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
    image_smoothed = cv2.medianBlur(image_gray, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    image_smoothed = cv2.morphologyEx(image_smoothed, cv2.MORPH_OPEN, kernel)
    image_smoothed = cv2.GaussianBlur(image_smoothed, (9, 9), 0)

    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))
    
    if hasattr(cv2, 'HOUGH_GRADIENT_ALT'):
        circles = cv2.HoughCircles(
            image_smoothed, cv2.HOUGH_GRADIENT_ALT, 
            minDist=min(h, w) * 0.8, dp=1.5, param1=300, param2=0.85, 
            minRadius=r_min, maxRadius=r_max
        )
        if circles is not None:
            cx, cy, r = circles[0][0]
            return int(np.round(cy)), int(np.round(cx)), int(np.round(r))

    return h // 2, w // 2, min(h, w) // 2


# ==========================================
# 4. Binary Mask Hough
# ==========================================
def find_circle_Binary_Mask(pil_img: PIL.Image.Image) -> tuple[int, int, int]:
    """
    Experimental approach generating a pure binary mask to feed into HoughCircles.
    """
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
    blur_k = min(31, min(h, w) // 2 | 1)
    image_smoothed = cv2.medianBlur(image_gray, blur_k)
    _, binary = cv2.threshold(image_smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary_blurred = cv2.GaussianBlur(binary, (15, 15), 0)

    r_min = int(0.25 * max(h, w))
    r_max = int(0.75 * min(h, w))
    
    circles = cv2.HoughCircles(
        binary_blurred, cv2.HOUGH_GRADIENT, 
        minDist=min(h, w) * 0.8, dp=1.0, param1=50, param2=15, 
        minRadius=r_min, maxRadius=r_max
    )
    if circles is not None:
        cx, cy, r = circles[0][0]
        return int(np.round(cy)), int(np.round(cx)), int(np.round(r))
        
    return h // 2, w // 2, min(h, w) // 2


# ==========================================
# 5. Grid Search Runner for Hough
# ==========================================
class AlgoB_Runner:
    """
    Runner class used to grid search HoughCircles parameters efficiently.
    Usage: run_all(val_set, algo=AlgoB_Runner(blur_size=21, param1=100, param2=0.8))
    """
    def __init__(self, blur_size, param1, param2):
        self.blur_size = blur_size
        self.param1 = param1
        self.param2 = param2

    def __call__(self, pil_img: PIL.Image.Image) -> tuple[int, int, int]:
        image_np = np.array(pil_img.convert("L"))
        h, w = image_np.shape

        cy_min, cy_max = int(0.3 * h), int(0.7 * h)
        cx_min, cx_max = int(0.3 * w), int(0.7 * w)
        r_min = int(0.25 * max(h, w))
        r_max = int(0.75 * min(h, w))

        blurred = cv2.GaussianBlur(image_np, (self.blur_size, self.blur_size), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        if hasattr(cv2, 'HOUGH_GRADIENT_ALT'):
            method = cv2.HOUGH_GRADIENT_ALT
        else:
            method = cv2.HOUGH_GRADIENT
            
        circles = cv2.HoughCircles(
            enhanced, method, dp=1.5, minDist=min(h, w),
            param1=self.param1, param2=self.param2,
            minRadius=r_min, maxRadius=r_max
        )

        if circles is not None:
            cx, cy, r = np.round(circles[0][0]).astype(int)
        else:
            cx, cy, r = w // 2, h // 2, int((r_min + r_max) / 2)

        cy = int(np.clip(cy, cy_min, cy_max))
        cx = int(np.clip(cx, cx_min, cx_max))
        r  = int(np.clip(r, r_min, r_max))

        return cy, cx, r
