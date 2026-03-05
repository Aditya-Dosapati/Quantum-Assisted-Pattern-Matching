"""FastAPI backend for Quantum-Assisted Pattern Matching."""

import io
import os
import sys
import base64

# Ensure local modules are importable regardless of CWD
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import json
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import load_models, DEVICE
from features import extract_clip_features
from similarity import ae_qip_similarity, edge_structure_similarity, color_hist_similarity
from detection import detect_grid_tiles, uniform_grid_split
from quantum import run_grover_search
import time

_STATIC_DIR = os.path.join(_BASE_DIR, "static")

app = FastAPI(title="Quantum Pattern Matching")

# Serve static files
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _img_to_b64(img_array: np.ndarray, fmt: str = "png") -> str:
    """Convert a numpy RGB image to a base64-encoded data URL."""
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(f".{fmt}", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64 data URL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/device")
async def get_device():
    return {"device": DEVICE}


@app.post("/api/analyze")
async def analyze(
    scene: UploadFile = File(...),
    target: UploadFile = File(...),
):
    """Run the full analysis pipeline and return JSON results."""
    pipeline_start = time.time()
    yolo_model, clip_model, clip_processor = load_models()

    # --- Read images ---
    scene_bytes = await scene.read()
    target_bytes = await target.read()
    scene_img = np.array(Image.open(io.BytesIO(scene_bytes)).convert("RGB"))
    target_img = np.array(Image.open(io.BytesIO(target_bytes)).convert("RGB"))
    target_crop = target_img
    scene_h, scene_w = scene_img.shape[:2]
    target_h, target_w = target_img.shape[:2]

    t_detection_start = time.time()
    # --- Detection strategies ---
    # A: YOLO
    scene_results = yolo_model(scene_img, conf=0.15)
    yolo_candidates = []
    for box in scene_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = scene_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        yolo_candidates.append((x1, y1, x2, y2, crop))

    # B: Grid tiles
    grid_candidates = detect_grid_tiles(scene_img)

    # C: Uniform grid
    h, w = scene_img.shape[:2]
    est_tiles = max(4, min(25, int(np.sqrt((w * h) / (200 * 200)))))
    est_rows = max(2, int(np.sqrt(est_tiles * h / (w + 1e-6))))
    est_cols = max(2, int(np.sqrt(est_tiles * w / (h + 1e-6))))
    uniform_candidates = uniform_grid_split(scene_img, est_rows, est_cols)

    # Choose best strategy
    if len(grid_candidates) >= 4:
        candidates = grid_candidates
        detection_method = "Grid Tile Detection (contour-based)"
    elif len(yolo_candidates) >= 2:
        candidates = yolo_candidates
        detection_method = "YOLO Object Detection"
    elif len(uniform_candidates) >= 4:
        candidates = uniform_candidates
        detection_method = f"Uniform Grid Split ({est_rows}×{est_cols})"
    elif len(yolo_candidates) >= 1:
        candidates = yolo_candidates
        detection_method = "YOLO Object Detection"
    else:
        candidates = uniform_grid_split(scene_img, 3, 3)
        detection_method = "Uniform Grid Split (3×3 fallback)"

    n_candidates = len(candidates)
    if n_candidates == 0:
        return {"error": "No candidate regions found in scene."}
    t_detection_end = time.time()

    t_similarity_start = time.time()
    # --- CLIP features ---
    candidate_feats = [extract_clip_features(c[-1], clip_model, clip_processor) for c in candidates]
    target_feat = extract_clip_features(target_crop, clip_model, clip_processor)

    # --- Negative anchors / noise floor ---
    noise_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    noise_feat = extract_clip_features(noise_img, clip_model, clip_processor)
    gray_img = np.full((224, 224, 3), 128, dtype=np.uint8)
    gray_feat = extract_clip_features(gray_img, clip_model, clip_processor)
    scene_feat = extract_clip_features(scene_img, clip_model, clip_processor)

    noise_scores = [float(ae_qip_similarity(f, noise_feat)) for f in candidate_feats]
    gray_scores = [float(ae_qip_similarity(f, gray_feat)) for f in candidate_feats]
    noise_floor = float(max(max(noise_scores), max(gray_scores)))

    cross_sims = []
    for i in range(len(candidate_feats)):
        for j in range(i + 1, len(candidate_feats)):
            cross_sims.append(float(ae_qip_similarity(candidate_feats[i], candidate_feats[j])))
    avg_cross_similarity = float(np.mean(cross_sims)) if cross_sims else 0.5

    # --- AE-QIP similarity ---
    similarity_scores = [float(ae_qip_similarity(f, target_feat)) for f in candidate_feats]
    best_classical = int(np.argmax(similarity_scores))
    best_score = float(similarity_scores[best_classical])
    t_similarity_end = time.time()

    raw_cosines = [float(np.dot(f, target_feat)) for f in candidate_feats]
    best_raw_cosine = raw_cosines[best_classical]

    scene_baseline = float(ae_qip_similarity(scene_feat, target_feat))
    baseline_gap = float(best_score - scene_baseline)
    noise_margin = float(best_score - noise_floor)
    cross_margin = float(best_score - avg_cross_similarity)

    if len(similarity_scores) > 1:
        other_scores = [s for i, s in enumerate(similarity_scores) if i != best_classical]
        mean_others = float(np.mean(other_scores))
        std_others = float(np.std(other_scores)) + 1e-8
        separation = float((best_score - mean_others) / std_others)
        score_gap = float(best_score - mean_others)
    else:
        mean_others = 0.0
        separation = 10.0
        score_gap = best_score

    # --- Pattern-absence detection ---
    pattern_absent = False
    rejection_reasons = []

    if noise_margin < 0.05:
        pattern_absent = True
        rejection_reasons.append(
            f"Best score ({best_score:.4f}) barely above noise floor "
            f"({noise_floor:.4f}), margin: {noise_margin:.4f} < 0.05"
        )
    if cross_margin < 0.03 and len(cross_sims) > 0:
        pattern_absent = True
        rejection_reasons.append(
            f"Best score ({best_score:.4f}) not above scene self-similarity "
            f"({avg_cross_similarity:.4f}), margin: {cross_margin:.4f} < 0.03"
        )
    if baseline_gap < 0.02:
        pattern_absent = True
        rejection_reasons.append(
            f"No localization — best candidate ({best_score:.4f}) ≈ full scene "
            f"({scene_baseline:.4f}), gap: {baseline_gap:.4f} < 0.02"
        )
    if len(similarity_scores) > 2 and separation < 2.0 and noise_margin < 0.10:
        pattern_absent = True
        rejection_reasons.append(
            f"No standout candidate (z-score: {separation:.2f} < 2.0) "
            f"and weak noise margin ({noise_margin:.4f} < 0.10)"
        )
    if len(similarity_scores) > 2:
        score_range = max(similarity_scores) - min(similarity_scores)
        if score_range < 0.02 and noise_margin < 0.08:
            pattern_absent = True
            rejection_reasons.append(
                f"All candidates nearly identical scores (range: {score_range:.4f} < 0.02), "
                f"weak noise margin ({noise_margin:.4f} < 0.08)"
            )

    if pattern_absent:
        return {
            "error": "Pattern NOT found in the scene.",
            "rejection_reasons": rejection_reasons,
            "diagnostics": {
                "best_score": round(best_score, 4),
                "noise_floor": round(noise_floor, 4),
                "noise_margin": round(noise_margin, 4),
                "avg_cross_similarity": round(avg_cross_similarity, 4),
                "scene_baseline": round(scene_baseline, 4),
            },
        }

    # --- Multi-match selection ---
    diff_threshold_pct = 0.5
    diff_threshold = diff_threshold_pct / 100.0
    ranked_by_score = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
    best_idx = ranked_by_score[0]
    second_idx = ranked_by_score[1] if len(ranked_by_score) > 1 else None
    best_score_pct = similarity_scores[best_idx] * 100.0
    second_score_pct = similarity_scores[second_idx] * 100.0 if second_idx is not None else None
    top_gap_pct = (best_score_pct - second_score_pct) if second_score_pct is not None else 100.0

    if top_gap_pct > diff_threshold_pct:
        matched_indices = [best_idx]
    else:
        matched_indices = [
            i for i, score in enumerate(similarity_scores)
            if ((best_score - score) * 100.0) <= diff_threshold_pct
        ]
    match_threshold = best_score - diff_threshold
    if not matched_indices:
        matched_indices = [best_classical]

    # --- Grover search ---
    t_quantum_start = time.time()
    shots = 1024
    grover_index, counts, qc, n_qubits, marked_state, iterations = run_grover_search(
        n_candidates, best_classical, shots=shots
    )
    t_quantum_end = time.time()

    # --- Edge & color similarity for best match ---
    best_crop = candidates[best_classical][-1]
    edge_sim = float(edge_structure_similarity(best_crop, target_crop))
    color_sim = float(color_hist_similarity(best_crop, target_crop))

    # --- Confidence score (weighted combination) ---
    confidence = float(0.50 * best_score + 0.25 * edge_sim + 0.25 * color_sim)

    # --- Quantum circuit diagram ---
    try:
        fig_qc, ax_qc = plt.subplots(figsize=(max(8, n_qubits * 2), max(3, n_qubits * 0.8)))
        qc.draw('mpl', ax=ax_qc)
        ax_qc.set_title("Grover's Circuit", fontsize=12, fontweight='bold', pad=10)
        circuit_b64 = _fig_to_b64(fig_qc)
        plt.close(fig_qc)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[WARN] Circuit drawing failed: {exc}")
        # Fallback: render text-based circuit as an image
        try:
            circuit_text = qc.draw('text').single_string() if hasattr(qc.draw('text'), 'single_string') else str(qc.draw('text'))
            fig_fb, ax_fb = plt.subplots(figsize=(max(10, n_qubits * 2.5), max(4, n_qubits * 1.2)))
            ax_fb.axis('off')
            ax_fb.text(0.02, 0.95, circuit_text, transform=ax_fb.transAxes,
                       fontsize=9, fontfamily='monospace', verticalalignment='top',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e1b4b', edgecolor='#6366f1', alpha=0.95))
            ax_fb.set_facecolor('#0a0e27')
            fig_fb.patch.set_facecolor('#0a0e27')
            ax_fb.set_title("Grover's Circuit (text)", fontsize=12, fontweight='bold', pad=10, color='white')
            circuit_b64 = _fig_to_b64(fig_fb)
            plt.close(fig_fb)
        except Exception as exc2:
            print(f"[WARN] Circuit text fallback also failed: {exc2}")
            circuit_b64 = None

    # --- Build candidate thumbnails (top 8) ---
    ranked_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)
    top_n = min(8, len(ranked_indices))
    candidate_thumbs = []
    for rank in range(top_n):
        idx = ranked_indices[rank]
        thumb = cv2.resize(candidates[idx][-1], (120, 120))
        candidate_thumbs.append({
            "rank": rank + 1,
            "index": idx,
            "score": round(similarity_scores[idx], 4),
            "image": _img_to_b64(thumb),
        })

    # --- Output image with bounding boxes ---
    output_img = scene_img.copy()
    for i, (cx1, cy1, cx2, cy2, _) in enumerate(candidates):
        if i in matched_indices:
            continue
        cv2.rectangle(output_img, (cx1, cy1), (cx2, cy2), (100, 100, 100), 1)
    for match_rank, idx in enumerate(matched_indices, start=1):
        x1, y1, x2, y2, _ = candidates[idx]
        thickness = 4 if idx == grover_index else 3
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(output_img, f"MATCH {match_rank}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- Charts ---
    # Chart 1: Grover measurement histogram
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    states = list(counts.keys())
    count_vals = list(counts.values())
    colors_grover = [
        '#10b981' if s == format(grover_index, f'0{len(states[0])}b') else '#6366f1'
        for s in states
    ]
    bars = ax1.bar(states, count_vals, color=colors_grover, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, count_vals):
        if val > max(count_vals) * 0.05:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(count_vals) * 0.02,
                     str(val), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e293b')
    ax1.set_xlabel("Quantum State", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Measurement Counts", fontsize=11, fontweight='bold')
    ax1.set_title("Grover's Algorithm — Measurement Distribution", fontsize=13, fontweight='bold', pad=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', rotation=45 if len(states) > 8 else 0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    fig1.tight_layout()
    chart1_b64 = _fig_to_b64(fig1)
    plt.close(fig1)

    # Chart 2: Candidate similarity scores
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cand_labels = [f"C{i}" for i in range(len(similarity_scores))]
    colors_sim = [
        '#10b981' if i in matched_indices
        else '#ef4444' if similarity_scores[i] < match_threshold
        else '#6366f1'
        for i in range(len(similarity_scores))
    ]
    bars2 = ax2.barh(cand_labels, similarity_scores, color=colors_sim, edgecolor='white', linewidth=0.8)
    for bar, score in zip(bars2, similarity_scores):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{score:.3f}", ha='left', va='center', fontsize=9, fontweight='bold', color='#1e293b')
    ax2.axvline(x=noise_floor, color='#94a3b8', linestyle=':', linewidth=1.2,
                label=f'Noise Floor ({noise_floor:.3f})')
    ax2.axvline(x=match_threshold, color='#ef4444', linestyle='--', linewidth=1.5,
                label=f'Min Match ({match_threshold:.3f})')
    ax2.set_xlabel("Similarity Score", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Candidate Region", fontsize=11, fontweight='bold')
    ax2.set_title("CLIP Similarity Scores per Candidate Region", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlim(0, min(1.05, max(similarity_scores) + 0.08))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    fig2.tight_layout()
    chart2_b64 = _fig_to_b64(fig2)
    plt.close(fig2)

    plt.close('all')

    # --- Timing ---
    pipeline_end = time.time()
    timing = {
        "detection_ms": round((t_detection_end - t_detection_start) * 1000),
        "similarity_ms": round((t_similarity_end - t_similarity_start) * 1000),
        "quantum_ms": round((t_quantum_end - t_quantum_start) * 1000),
        "total_ms": round((pipeline_end - pipeline_start) * 1000),
    }

    # --- Grover counts for frontend chart ---
    sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_states = sorted_states[:min(8, len(sorted_states))]

    return JSONResponse(content=json.loads(json.dumps({
        "detection_method": detection_method,
        "n_candidates": n_candidates,
        "best_classical": best_classical,
        "best_score": round(best_score, 4),
        "matched_indices": matched_indices,
        "n_matches": len(matched_indices),
        "grover_index": grover_index,
        "n_qubits": n_qubits,
        "marked_state": marked_state,
        "grover_iterations": iterations,
        "shots": shots,
        "candidate_thumbs": candidate_thumbs,
        "similarity_scores": [round(s, 4) for s in similarity_scores],
        "output_image": _img_to_b64(output_img),
        "scene_image": _img_to_b64(scene_img),
        "target_image": _img_to_b64(target_crop),
        "chart_grover": chart1_b64,
        "chart_similarity": chart2_b64,
        "chart_circuit": circuit_b64,
        "top_gap_pct": round(top_gap_pct, 2),
        "edge_similarity": round(edge_sim, 4),
        "color_similarity": round(color_sim, 4),
        "confidence": round(confidence, 4),
        "timing": timing,
        "image_info": {
            "scene_width": scene_w,
            "scene_height": scene_h,
            "target_width": target_w,
            "target_height": target_h,
            "scene_pixels": scene_w * scene_h,
            "target_pixels": target_w * target_h,
        },
        "diagnostics": {
            "best_score": round(best_score, 4),
            "best_raw_cosine": round(best_raw_cosine, 4),
            "noise_floor": round(noise_floor, 4),
            "noise_margin": round(noise_margin, 4),
            "avg_cross_similarity": round(avg_cross_similarity, 4),
            "cross_margin": round(cross_margin, 4),
            "scene_baseline": round(scene_baseline, 4),
            "baseline_gap": round(baseline_gap, 4),
            "mean_others": round(mean_others, 4),
            "score_gap": round(score_gap, 4),
            "separation": round(separation, 2),
            "all_scores": [round(s, 4) for s in similarity_scores],
        },
        "quantum_info": {
            "n_candidates": n_candidates,
            "n_qubits": n_qubits,
            "state_space": 2 ** n_qubits,
            "marked_state": marked_state,
            "iterations": iterations,
            "shots": shots,
            "top_states": [{"state": s, "count": c} for s, c in top_states],
        },
    }, cls=_NumpyEncoder)))


if __name__ == "__main__":
    import uvicorn
    print("Loading models on startup...")
    load_models()
    print("Models loaded. Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
