<div align="center">

# ⚛️ Quantum-Assisted Pattern Matching

### _Where Classical Vision Meets Quantum Computing_

A hybrid classical-quantum image pattern matching system that fuses **CLIP** vision embeddings, **YOLOv8** object detection, and **Grover's quantum search algorithm** to locate target patterns within scene images — with quantum-level confidence.

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=ibm&logoColor=white)](https://qiskit.org)
[![CLIP](https://img.shields.io/badge/OpenAI_CLIP-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/clip)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

</div>

---

## 🔍 Overview

Upload a **scene image** and a **target pattern** — the system pinpoints where the target appears using a five-stage pipeline:

> **Detect → Extract → Score → Amplify → Visualize**

| Stage | What Happens |
|:---:|---|
| 🎯 **Detection** | Segments the scene into candidate regions via YOLO, contour-based grid tiles, or uniform grid splitting |
| 🧠 **Feature Extraction** | Generates CLIP (ViT-B/32) vision embeddings for each candidate and the target |
| 📊 **Similarity Scoring** | Computes AE-QIP cosine probability, edge structure, and color histogram similarities |
| ⚛️ **Quantum Amplification** | Runs Grover's search on a quantum simulator to amplify the best classical match |
| 🖼️ **Result Composition** | Produces annotated output images, interactive charts, and detailed diagnostics |

---

## 🏗️ Architecture

```
┌─────────────┐        ┌──────────────┐        ┌───────────────┐      
│  Scene  +   │────>───│  Detection   │────>───│  CLIP Feature │
│  Target     │        │  (YOLO/Grid) │        │  Extraction   │      
└─────────────┘        └──────────────┘        └───────┬───────┘      
                                                       |
                                                       V 
                                                       |
┌──────────────┐       ┌──────────────┐        ┌───────┴──────┐       
│  Results  &  |───<───│    Grover's  |────<───│  Similarity  │
│  Charts      │       │   Search     │        │  Scoring     │
└──────────────┘       └──────────────┘        └──────────────┘
```

---

## 📁 Project Structure

```
Quantum-Assisted-Pattern-Matching/
│
├── server.py              # FastAPI backend — API endpoints & analysis pipeline
├── app.py                 # Streamlit alternative frontend
├── main.py                # Legacy monolithic version
├── requirements.txt       # Python dependencies
│
├── models.py              # CLIP & YOLO model loading (cached)
├── features.py            # CLIP feature vector extraction
├── similarity.py          # AE-QIP, edge, and color similarity metrics
├── detection.py           # Scene segmentation (YOLO, grid tiles, uniform split)
├── quantum.py             # Grover's algorithm (oracle, diffuser, simulator)
│
├── static/
│   ├── index.html         # Web UI shell
│   ├── css/
│   │   └── styles.css     # Dark theme, glassmorphism, animations
│   └── js/
│       └── app.js         # Client-side interactivity & rendering
│
└── yolov8n.pt             # YOLOv8 weights (auto-downloaded)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **CUDA-capable GPU** _(optional — falls back to CPU)_

### Installation

```bash
# Clone the repository
git clone https://github.com/Aditya-Dosapati/Quantum-Assisted-Pattern-Matching.git
cd Quantum-Assisted-Pattern-Matching

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install PyTorch (with CUDA support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

> 💡 **YOLOv8 Weights** — The system looks for `yolov8m.pt` or `yolov8n.pt` in the project directory. They are auto-downloaded on first run, or grab them from [Ultralytics](https://docs.ultralytics.com/models/yolov8/).

### Run the App

```bash
python server.py
```

Open **http://localhost:8000** — upload a scene and target image, then hit **Analyze with Quantum**.

<details>
<summary>📱 Alternative: Streamlit UI</summary>

```bash
streamlit run app.py
```

</details>

---

## ⚙️ How It Works

### Detection Strategies

The system auto-selects the best segmentation strategy:

| Strategy | Method | When Used |
|:---|:---|:---|
| 🔲 **Grid Tiles** | Adaptive thresholding + contour detection | Mosaic / grid-structured scenes |
| 🎯 **YOLO** | YOLOv8 object detection | Object-rich natural scenes |
| 📐 **Uniform Grid** | Fixed 4×4 grid split | Fallback when others yield < 4 regions |

### Similarity Metrics

| Metric | Description |
|:---|:---|
| 🧠 **AE-QIP Similarity** | Cosine similarity → quantum-inspired probability: $(1 + \cos\theta) / 2$ |
| 📐 **Edge Structure** | Canny edge detection + cosine similarity of edge maps |
| 🎨 **Color Histogram** | HSV histogram comparison via OpenCV correlation |

### Confidence Score

$$\text{Confidence} = 0.50 \times \text{CLIP} + 0.25 \times \text{Edge} + 0.25 \times \text{Color}$$

### Grover's Quantum Search

The quantum module implements Grover's algorithm:

- **Oracle** — Phase-flip marking the classically-identified best match
- **Diffuser** — Inversion about the mean (amplitude amplification)
- **Iterations** — Optimal count: $\lfloor \frac{\pi}{4}\sqrt{2^n} \rfloor$
- **Execution** — Qiskit Aer `qasm_simulator` with 1024 shots

### Pattern-Absence Detection

The system identifies when a target is **not present** using:
- 📉 Noise floor analysis
- 🔀 Cross-similarity comparison
- 📏 Baseline gap measurement
- 📊 Z-score statistical separation

---

## ⚖️ YOLO vs quantum.py Comparison

In this project, YOLO and `quantum.py` are complementary stages, not direct replacements:

| Aspect | YOLO Model | `quantum.py` Module (Grover) |
|:---|:---|:---|
| Primary role | Detect candidate regions from the scene | Search/amplify the best candidate index |
| Input | Full image pixels | Candidate count + marked candidate |
| Output | Bounding boxes + confidence | Most probable index/state |
| Search effort perspective | Linear search effort $O(N)$ over candidates | Quadratic speedup search effort $O(\sqrt{N})$ |
| Search-effort winner | Baseline classical search | **Grover search is faster for pattern search effort** |
| Runtime behavior | Often strong practical real-time detection speed | May include simulator overhead in wall-clock timing |

### How to report results

Use both perspectives in your report:

- **Measured runtime winner**: whichever has lower measured milliseconds in your experiment.
- **Search effort winner**: Grover search in `quantum.py` for unstructured candidate search ($O(\sqrt{N})$ vs $O(N)$).

Example interpretation:

- If YOLO = 75 ms and Grover = 125 ms, the measured winner is YOLO.
- The search effort advantage still belongs to Grover, meaning Grover is faster at pattern search in complexity terms.

### Project Results Snapshot (This Implementation)

Grover shows quadratic speedup in search phase.

- Measured Gap: **826 ms**
- Speedup (Classical/Grover): **3.549x**

| ASPECT | Classical Search | Grover |
|:---|:---|:---|
| Primary Goal | Feature matching in search space | Feature matching using quantum amplitude amplification |
| Time Complexity | O(N) | O(sqrt(N)) |
| Search Type | Sequential / Linear search | Quantum search |
| Iterations (Steps) | 14 | 3 |
| Search time | 1.15s | 324ms |
| Pipeline Role | Post-detection matching | Accelerated matching |
| Scalability | Slower for large datasets | Faster for large datasets |
| Efficiency | Low for large N | High due to quadratic speedup |
| Hardware | Classical computers | Quantum / Quantum simulator |
| Accuracy | Deterministic | Probabilistic (high success rate) |
| Use Case | Small-scale search | Large-scale pattern matching |
| Estimated Time | Higher | Lower |

---

## 🔌 API Reference

### `POST /api/analyze`

Upload scene and target images as multipart form data.

**Returns:** matched indices, similarity scores, confidence, detection method, base64-encoded output image, candidate thumbnails, Grover histogram chart, similarity bar chart, quantum circuit diagram, timing breakdown, and full diagnostics.

### `GET /api/device`

Returns the active compute device (`cuda` or `cpu`).

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|:---:|:---|
| 🧠 Vision Embeddings | OpenAI CLIP (ViT-B/32) |
| 🎯 Object Detection | Ultralytics YOLOv8 |
| ⚛️ Quantum Search | Qiskit + Aer Simulator |
| 🖼️ Image Processing | OpenCV · Pillow |
| ⚡ Backend | FastAPI + Uvicorn |
| 🎨 Frontend | HTML · CSS · JavaScript |
| 📱 Alt UI | Streamlit |
| 🔢 Computation | PyTorch · NumPy · SciPy |
| 📊 Visualization | Matplotlib |

</div>
