# Quantum-Assisted Pattern Matching

A hybrid classical-quantum image pattern matching system that combines **CLIP** vision embeddings, **YOLOv8** object detection, and **Grover's quantum search algorithm** to locate a target pattern within a scene image.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Qiskit](https://img.shields.io/badge/Qiskit-Quantum-6929C4?logo=ibm)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF)

---

## Overview

The system takes two inputs — a **scene image** and a **target pattern** — and identifies where the target appears within the scene. It uses a multi-stage pipeline:

1. **Detection** — Segments the scene into candidate regions using YOLO object detection, contour-based grid tile detection, or uniform grid splitting
2. **Feature Extraction** — Extracts CLIP vision embeddings from each candidate region and the target
3. **Similarity Scoring** — Computes AE-QIP (cosine-based quantum-inspired probability), edge structure, and color histogram similarities
4. **Quantum Amplification** — Runs Grover's search algorithm on a quantum simulator to amplify the best classical match
5. **Result Composition** — Generates annotated output images, charts, and detailed diagnostics

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Scene +    │───▶│  Detection   │───▶│  CLIP Feature │───▶│  Similarity  │
│  Target     │    │  (YOLO/Grid) │    │  Extraction   │    │  Scoring     │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
                                                                   │
                   ┌──────────────┐    ┌───────────────┐           │
                   │   Results    │◀───│   Grover's    │◀──────────┘
                   │   & Charts   │    │   Search      │
                   └──────────────┘    └───────────────┘
```

## Project Structure

```
├── server.py          # FastAPI backend — API endpoints & analysis pipeline
├── app.py             # Streamlit alternative frontend
├── models.py          # CLIP & YOLO model loading (cached)
├── features.py        # CLIP feature vector extraction
├── similarity.py      # AE-QIP, edge, and color similarity metrics
├── detection.py       # Scene segmentation (YOLO, grid tiles, uniform split)
├── quantum.py         # Grover's algorithm (oracle, diffuser, simulator)
├── static/
│   └── index.html     # Web UI (dark theme, glassmorphism)
├── .streamlit/
│   └── config.toml    # Streamlit configuration
├── main.py            # Legacy monolithic version
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, falls back to CPU)

### Installation

```bash
git clone https://github.com/Aditya-Dosapati/Quantum-Assisted-Pattern-Matching.git
cd Quantum-Assisted-Pattern-Matching
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics transformers qiskit qiskit-aer
pip install opencv-python numpy Pillow matplotlib
pip install fastapi uvicorn python-multipart streamlit
```

### YOLOv8 Weights

The system looks for YOLOv8 weights (`yolov8m.pt` or `yolov8n.pt`) in the project directory and common paths. You can download them from [Ultralytics](https://docs.ultralytics.com/models/yolov8/).

## Usage

### FastAPI Web App (Recommended)

```bash
python server.py
```

Open **http://localhost:8000** in your browser. Upload a scene image and a target pattern, then click **Analyze with Quantum**.

### Streamlit App

```bash
streamlit run app.py
```

## How It Works

### Detection Strategies

The system automatically selects the best detection strategy:

| Strategy | Method | When Used |
|----------|--------|-----------|
| **Grid Tiles** | Adaptive thresholding + contour detection | Mosaic/grid-structured scenes |
| **YOLO** | YOLOv8 object detection | Object-rich scenes |
| **Uniform Grid** | Fixed 4×4 grid split | Fallback when others yield < 4 regions |

### Similarity Metrics

- **AE-QIP Similarity** — Cosine similarity mapped to quantum-inspired probability: $(1 + \cos\theta) / 2$
- **Edge Structure** — Canny edge detection + cosine similarity of edge maps
- **Color Histogram** — HSV histogram comparison using OpenCV correlation

### Confidence Score

$$\text{Confidence} = 0.50 \times \text{CLIP} + 0.25 \times \text{Edge} + 0.25 \times \text{Color}$$

### Grover's Search

The quantum module implements Grover's algorithm with:
- Phase-flip oracle marking the classically-identified best match
- Diffusion operator (inversion about the mean)
- Optimal iterations: $\lfloor \frac{\pi}{4}\sqrt{2^n} \rfloor$
- Execution on Qiskit Aer `qasm_simulator` (1024 shots)

### Pattern-Absence Detection

The system detects when a target pattern is **not present** in the scene using:
- Noise floor analysis
- Cross-similarity comparison
- Baseline gap measurement
- Z-score statistical separation

## API

### `POST /api/analyze`

Upload scene and target images as multipart form data.

**Response** includes: matched indices, similarity scores, confidence, detection method, base64-encoded output image, candidate thumbnails, charts (Grover histogram, similarity bars), timing breakdown, quantum circuit diagram, and diagnostics.

### `GET /api/device`

Returns the compute device info (CUDA/CPU).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vision Embeddings | OpenAI CLIP (ViT-B/32) |
| Object Detection | Ultralytics YOLOv8 |
| Quantum Search | Qiskit + Aer Simulator |
| Image Processing | OpenCV, Pillow |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Alternative UI | Streamlit |
| Computation | PyTorch, NumPy |
| Visualization | Matplotlib |
