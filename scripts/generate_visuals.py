import os
import json
from pathlib import Path
import PIL.Image
import PIL.ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Assuming the script is run from the root of the project
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.gradient import find_circle_gradient as find_circle_A
from src.algorithms.hough import find_circle_B

def generate_beautiful_visuals():
    img_dir = Path("data/circle")
    val_dir = img_dir / "val"
    
    # Check if dataset exists
    if not val_dir.exists():
        print(f"Error: Dataset not found at {val_dir}. Please download it first.")
        return

    # Load ground truths
    gt_file = img_dir / "val.json"
    with open(gt_file, "r") as f:
        data = json.load(f)
        ground_truths = {item['path']: item for item in data}

    # Pick 3 interesting images from different difficulty levels
    # Let's find one image from diff 0, one from 40-60, one from 80-90
    sample_images = []
    difficulties_target = [('val/0', 0), ('val/50', 50), ('val/90', 90)]
    for target_prefix, _ in difficulties_target:
        for p in ground_truths.keys():
            if p.startswith(target_prefix):
                sample_images.append(p)
                break
    
    # Fallback if we couldn't find exactly those
    if len(sample_images) < 3:
        sample_images = list(ground_truths.keys())[:3]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18), facecolor='#1e1e2e')
    fig.suptitle('Eye Circle Detection: Algorithm Comparison', fontsize=26, color='white', y=0.95, fontweight='bold')
    
    for idx, img_path in enumerate(sample_images):
        full_path = img_dir / img_path
        pil_img = PIL.Image.open(full_path).convert("RGB")
        img_np = np.array(pil_img)

        # Ground Truth
        gt = ground_truths[img_path]
        
        # Predictions
        cy_A, cx_A, r_A = find_circle_A(pil_img)
        cy_B, cx_B, r_B = find_circle_B(pil_img)

        difficulty = img_path.split('/')[1] if '/' in img_path else "Unknown"

        # Col 0: Ground Truth
        ax_gt = axes[idx, 0]
        ax_gt.imshow(img_np)
        ax_gt.axis('off')
        if idx == 0: ax_gt.set_title("Ground Truth", color='#a6e3a1', fontsize=20, pad=15)
        ax_gt.text(0.05, 0.95, f"Diff: {difficulty}", transform=ax_gt.transAxes, color='white', fontsize=16, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        gt_circle = Circle((gt['cx'], gt['cy']), gt['r'], fill=False, edgecolor='#a6e3a1', linewidth=4)
        ax_gt.add_patch(gt_circle)

        # Col 1: Algorithm A
        ax_a = axes[idx, 1]
        ax_a.imshow(img_np)
        ax_a.axis('off')
        if idx == 0: ax_a.set_title("Algorithm A (Gradient)", color='#f5c2e7', fontsize=20, pad=15)
        a_circle = Circle((cx_A, cy_A), r_A, fill=False, edgecolor='#f5c2e7', linewidth=4)
        ax_a.add_patch(a_circle)

        # Col 2: Algorithm B
        ax_b = axes[idx, 2]
        ax_b.imshow(img_np)
        ax_b.axis('off')
        if idx == 0: ax_b.set_title("Algorithm B (Hough)", color='#89b4fa', fontsize=20, pad=15)
        b_circle = Circle((cx_B, cy_B), r_B, fill=False, edgecolor='#89b4fa', linewidth=4)
        ax_b.add_patch(b_circle)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    output_path = "algorithm_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Beautiful visualization saved to {output_path}!")

if __name__ == "__main__":
    generate_beautiful_visuals()
