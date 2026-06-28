import json
from pathlib import Path
from typing import TypedDict
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import PIL.Image

# Types defining a single result and a result with evaluation metrics
class Result(TypedDict):
    path: str  # Relative path to the image
    h: int     # Image height
    w: int     # Image width
    cy: int    # Y coordinate of the center
    cx: int    # X coordinate of the center
    r: int     # Circle radius
    duration: float  # Algorithm execution time in seconds

class ResultWithComparison(Result):
    c_dist: float
    r_diff: float
    iou: float


def load_dataset_paths(img_dir: Path) -> tuple[list[Path], list[Path]]:
    """Loads and sorts paths for training and validation images."""
    train_set = sorted((img_dir / "train").rglob("*.webp"))
    val_set = sorted((img_dir / "val").rglob("*.webp"))
    return train_set, val_set


def load_ground_truths(img_dir: Path) -> dict[Path, dict]:
    """Loads ground truths from JSON files into a convenient dictionary."""
    ground_truths = {}
    for part in ("train", "val", "test"):
        json_path = img_dir / f"{part}.json"
        if not json_path.exists():
            continue
        with json_path.open() as f:
            for result in json.load(f):
                ground_truths[img_dir / result["path"]] = result
    return ground_truths


class EyeDataset(Dataset):
    """
    PyTorch Dataset that loads images using generated "pseudo-ground-truths"
    or from a provided dataframe / CSV file.
    """
    def __init__(self, csv_file: str, img_size: int = 128):
        self.labels = pd.read_csv(csv_file)
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        row = self.labels.iloc[idx]
        image_path = row["image_path"]
        image = PIL.Image.open(image_path).convert("RGB")
        w, h = image.size
        
        image_transformed = self.transforms(image)

        # Normalizing labels so the model learns faster in the [0, 1] range
        cx_norm = float(row["cx"]) / w
        cy_norm = float(row["cy"]) / h
        r_norm = float(row["r"]) / max(h, w)

        labels_tensor = torch.tensor([cy_norm, cx_norm, r_norm], dtype=torch.float32)

        return image_transformed, labels_tensor


class ValDataset(Dataset):
    """Validation dataset using the ground_truths dictionary."""
    def __init__(self, val_paths: list[Path], ground_truths: dict[Path, dict], img_size: int = 128):
        self.paths = val_paths
        self.gt = ground_truths
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        w, h = image.size

        image_transformed = self.transforms(image)
        data = self.gt[img_path]

        cy_norm = float(data["cy"]) / h
        cx_norm = float(data["cx"]) / w
        r_norm = float(data["r"]) / max(h, w)

        labels_tensor = torch.tensor([cy_norm, cx_norm, r_norm], dtype=torch.float32)

        return image_transformed, labels_tensor
