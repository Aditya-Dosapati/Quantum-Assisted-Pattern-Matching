"""CLIP feature extraction utilities."""

import cv2
import numpy as np
import torch
from PIL import Image

from models import DEVICE


def extract_clip_features(img_crop, clip_model, clip_processor):
    """Extract normalised CLIP image features from a BGR/RGB crop."""
    pil_img = Image.fromarray(cv2.resize(img_crop, (224, 224)))
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        if isinstance(features, torch.Tensor):
            vec = features.detach().cpu().numpy().flatten()
        else:
            vec = features.pooler_output.detach().cpu().numpy().flatten()

    return vec / (np.linalg.norm(vec) + 1e-8)
