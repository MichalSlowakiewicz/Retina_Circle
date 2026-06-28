import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def get_efficientnet_model(pretrained: bool = True) -> nn.Module:
    """Returns an EfficientNet_B0 model with its classifier layer modified to output 3 values (cy, cx, r)."""
    if pretrained:
        weights = models.EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None
        
    model = models.efficientnet_b0(weights=weights)
    
    # Replacing the final Linear layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)
    
    return model

def find_circle_nn(pil_img: PIL.Image.Image, model: nn.Module, device: torch.device, img_size: int = 128) -> tuple[int, int, int]:
    """
    Performs inference using the trained EfficientNet model. Returns the center coordinates 
    and radius scaled back to the image's original dimensions.
    """
    model.eval()
    image_np = np.array(pil_img)
    h, w = image_np.shape[:2]

    # Transformations must be identical to those used during training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.squeeze(0).cpu().numpy()

    # Denormalizing the prediction to recover original coordinate values
    cy = np.clip(prediction[0], 0, 1) * h
    cx = np.clip(prediction[1], 0, 1) * w
    r = np.clip(prediction[2], 0, max(w, h)) * max(h, w)

    return int(np.round(cy)), int(np.round(cx)), int(np.round(r))
