"""Model loading for YOLO and CLIP."""

import os
import functools
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def load_models():
    """Load YOLO and CLIP models, searching common paths for YOLO weights."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preferred_model = os.getenv("YOLO_MODEL", "yolov8n.pt")
    lightweight_mode = os.getenv("LIGHTWEIGHT_MODE", "false").lower() in {"1", "true", "yes", "on"}
    search_paths = [
        os.path.join(script_dir, preferred_model),
        os.path.join(script_dir, "..", preferred_model),
        preferred_model,
        os.path.join(script_dir, "yolov8n.pt"),
        os.path.join(script_dir, "..", "yolov8n.pt"),
        os.path.join(script_dir, "yolov8m.pt"),
        os.path.join(script_dir, "..", "yolov8m.pt"),
        "yolov8m.pt",
        "yolov8n.pt",
    ]
    model_path = preferred_model  # fallback (will auto-download)
    for p in search_paths:
        if os.path.exists(p):
            model_path = os.path.abspath(p)
            break
    yolo = YOLO(model_path)
    if lightweight_mode:
        clip_model = None
        clip_processor = None
    else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", low_cpu_mem_usage=True).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return yolo, clip_model, clip_processor
