import json
import os
import threading
import time
import textwrap

import cv2
import matplotlib
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

from backend_emotion import emotion_detector
from eye_tracking.tracker import QuadrantEyeTracker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RAG.Main import generate_rag_answer_from_json

app = Flask(__name__, template_folder="templates", static_folder="static")

_gaze = "NO_FACE"
_gaze_lock = threading.Lock()
_cam_ready = False
_frame_count = 0
_cam_enabled = False
_session_gaze_points = []
_session_start_time = None
_session_blink_start = 0


def get_gaze():
    with _gaze_lock:
        return _gaze


def set_gaze(val):
    global _gaze
    with _gaze_lock:
        _gaze = val


# Images
IMAGE_DIR = os.path.join("static", "images")
IMAGE_EMOTION_MAP_PATH = os.path.join("static", "image_emotions.json")
_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}


def _normalize_image_emotion_map(data):
    if not isinstance(data, dict):
        print("[Images] Emotion map must be a JSON object")
        return {}

    normalized = {}

    for key, value in data.items():
        if isinstance(value, list):
            emotion = str(key).lower()
            for image_name in value:
                normalized[str(image_name)] = emotion
        else:
            normalized[str(key)] = str(value).lower()

    return normalized


def _group_images_by_emotion(image_emotions):
    grouped = {}
    for image_name, emotion in sorted(image_emotions.items()):
        emotion_key = str(emotion or "unknown").lower()
        grouped.setdefault(emotion_key, []).append(str(image_name))

    return dict(sorted(grouped.items()))


def _load_image_emotion_map():
    if not os.path.exists(IMAGE_EMOTION_MAP_PATH):
        return {}

    try:
        with open(IMAGE_EMOTION_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Images] Failed to read emotion map: {exc}")
        return {}

    return _normalize_image_emotion_map(data)


def _save_image_emotion_map(mapping):
    with open(IMAGE_EMOTION_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(_group_images_by_emotion(mapping), f, indent=4, sort_keys=True)


def _load_images():
    if not os.path.isdir(IMAGE_DIR):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        return [], {}

    files = sorted(
        f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1].lower() in _EXTS
    )
    emotion_map = _load_image_emotion_map()
    image_emotions = {f: emotion_map.get(f, "unknown") for f in files}

    if emotion_map != image_emotions:
        _save_image_emotion_map(image_emotions)

    return files, image_emotions


def refresh_image_catalog():
    global IMAGE_LIST, IMAGE_EMOTIONS
    IMAGE_LIST, IMAGE_EMOTIONS = _load_images()
    return IMAGE_LIST, IMAGE_EMOTIONS


def _safe_pdf_text(value):
    if value is None:
        return "-"
    return str(value).replace("\u2014", "-")


def _wrap_pdf_text(text, width=95):
    lines = []
    for paragraph in str(text or "").splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines or [""]


def _build_session_pdf(pdf_path, summary, heatmap_full_path="", scatter_full_path=""):
    metadata = summary.get("metadata", {})
    interactions = summary.get("interactions", [])
    rag_analysis = summary.get("rag_analysis", {})
    session_id = _safe_pdf_text(metadata.get("session_id", "unknown"))
    duration = metadata.get("total_duration_seconds", 0)
    total_blinks = metadata.get("total_blinks", 0)
    total_images = metadata.get("total_images_interacted", 0)
    match_count = sum(1 for item in interactions if item.get("matched"))

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
        ax.axis("off")
        ax.text(0, 1.0, "Session Report", fontsize=22, fontweight="bold", va="top")
        ax.text(0, 0.955, f"Session ID: {session_id}", fontsize=11, color="#444444", va="top")

        y = 0.905
        summary_lines = [
            f"Duration: {duration:.2f} seconds",
            f"Images viewed: {total_images}",
            f"Emotion matches: {match_count}",
            f"Total blinks: {total_blinks}",
        ]
        for line in summary_lines:
            ax.text(0, y, line, fontsize=11, va="top")
            y -= 0.032

        y -= 0.02
        ax.text(0, y, "RAG Analysis", fontsize=14, fontweight="bold", va="top")
        y -= 0.035
        for line in _wrap_pdf_text(_safe_pdf_text(rag_analysis.get("answer", "No analysis available."))):
            ax.text(0, y, line, fontsize=10.5, va="top")
            y -= 0.022
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
                ax.axis("off")
                y = 0.94

        y -= 0.02
        ax.text(0, y, "Interaction Details", fontsize=14, fontweight="bold", va="top")
        y -= 0.035
        if not interactions:
            ax.text(0, y, "No image interactions were recorded for this session.", fontsize=10.5, va="top")
        else:
            for index, item in enumerate(interactions, start=1):
                row = (
                    f"{index}. {_safe_pdf_text(item.get('image'))} | "
                    f"Target: {_safe_pdf_text(item.get('target_emotion'))} | "
                    f"User: {_safe_pdf_text(item.get('user_emotion'))} | "
                    f"Match: {'Yes' if item.get('matched') else 'No'} | "
                    f"Gaze: {_safe_pdf_text(item.get('gaze_region'))} | "
                    f"Time: {item.get('duration_seconds', 0):.2f}s | "
                    f"Views: {item.get('views', 0)}"
                )
                for line in _wrap_pdf_text(row, width=100):
                    ax.text(0, y, line, fontsize=9.5, va="top")
                    y -= 0.02
                    if y < 0.08:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                        fig = plt.figure(figsize=(8.27, 11.69))
                        ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
                        ax.axis("off")
                        ax.text(0, 0.97, "Interaction Details (continued)", fontsize=14, fontweight="bold", va="top")
                        y = 0.93
                y -= 0.01

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        plot_paths = [
            ("Gaze Heatmap", heatmap_full_path),
            ("Gaze Scatter Plot", scatter_full_path),
        ]
        for title, plot_path in plot_paths:
            if not plot_path or not os.path.exists(plot_path):
                continue
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_axes([0.06, 0.08, 0.88, 0.84])
            ax.axis("off")
            ax.set_title(title, fontsize=16, pad=16)
            ax.imshow(plt.imread(plot_path))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


IMAGE_LIST, IMAGE_EMOTIONS = _load_images()
print(f"[Images] {len(IMAGE_LIST)} images loaded")


# Tracker
tracker = QuadrantEyeTracker(smoothing_frames=7)


# Camera thread
def _camera_thread():
    global _cam_ready, _frame_count, _cam_enabled, _session_gaze_points

    cap = None
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[Camera] Opened index {idx}")
            break

    if not cap or not cap.isOpened():
        print("[Camera] ERROR: no camera found")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_detector.start()
    _cam_ready = True
    print("[Camera] Ready - waiting for session to start")

    while True:
        if not _cam_enabled:
            set_gaze("NO_FACE")
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        _frame_count += 1
        frame = cv2.flip(frame, 1)

        res = tracker.process(frame)
        if isinstance(res, dict):
            if res.get("blink"):
                set_gaze("BLINK")
            else:
                set_gaze(res["region"])
                if tracker.calibrated:
                    _session_gaze_points.append((res["x"], res["y"]))
        else:
            set_gaze(res)

        emotion_detector.push_frame(frame)


threading.Thread(target=_camera_thread, daemon=True).start()


# Routes
@app.route("/")
def index():
    refresh_image_catalog()
    return render_template("gaze_viewer.html", image_emotions=IMAGE_EMOTIONS)


@app.route("/gaze/stream")
def gaze_stream():
    def gen():
        prev = None
        while True:
            g = get_gaze()
            if g != prev:
                yield f"data: {g}\n\n"
                prev = g
            time.sleep(0.025)

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/images")
def list_images():
    refresh_image_catalog()
    return jsonify(images=IMAGE_LIST, emotions=IMAGE_EMOTIONS)


@app.route("/emotion")
def emotion():
    emo, conf = emotion_detector.get_emotion()
    return jsonify(emotion=emo, confidence=conf)


@app.route("/blinks")
def get_blinks():
    return jsonify(blinks=tracker.total_blinks)


# Calibration
@app.route("/calibrate", methods=["POST"])
def calibrate():
    tracker.reset_calibration()
    tracker.start_calibration()
    return jsonify(ok=True)


@app.route("/calibrate/status")
def calibrate_status():
    return jsonify(tracker.calibration_status())


# Camera on/off
@app.route("/camera/start", methods=["POST"])
def camera_start():
    global _cam_enabled, _session_start_time, _session_blink_start
    _cam_enabled = True
    _session_start_time = time.time()
    _session_blink_start = tracker.total_blinks
    print("[Camera] Enabled - session started")
    return jsonify(ok=True)


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    global _cam_enabled
    _cam_enabled = False
    set_gaze("NO_FACE")
    print("[Camera] Disabled - session ended")
    return jsonify(ok=True)


# Generate report and plots
@app.route("/api/stop_session", methods=["POST"])
def stop_session():
    global _session_start_time, _session_gaze_points

    frontend_data = request.json
    refresh_image_catalog()
    duration = time.time() - _session_start_time if _session_start_time else 0
    session_id = f"session_{int(time.time())}"
    session_blinks = max(tracker.total_blinks - _session_blink_start, 0)

    os.makedirs("static/results", exist_ok=True)

    rag_documents = []
    if frontend_data:
        for img_name, data in frontend_data.items():
            if data.get("views", 0) > 0:
                # Always use image_emotions.json as the authoritative target emotion
                target_emotion = IMAGE_EMOTIONS.get(img_name) or data.get("imgEmo", "unknown")
                if not target_emotion:
                    target_emotion = "unknown"
                time_s = data.get("viewMs", 0) / 1000.0
                match_str = (
                    "successfully matched"
                    if data.get("viewerEmo") == target_emotion and target_emotion != "unknown"
                    else "did not match"
                )

                text_block = (
                    f"During user session '{session_id}', the participant viewed the image "
                    f"'{img_name}' located in the '{data.get('lastGaze', 'unknown')}' screen "
                    f"quadrant. They spent {time_s:.2f} seconds looking at it. The target "
                    f"emotion to copy was '{target_emotion}'. The user's exhibited facial "
                    f"expression was '{data.get('viewerEmo')}', which {match_str} the target "
                    f"emotion."
                )

                rag_documents.append(
                    {
                        "image": img_name,
                        "target_emotion": target_emotion,
                        "user_emotion": data.get("viewerEmo"),
                        "matched": data.get("viewerEmo") == target_emotion and target_emotion != "unknown",
                        "gaze_region": data.get("lastGaze"),
                        "duration_seconds": round(time_s, 2),
                        "views": data.get("views"),
                        "rag_embedding_text": text_block,
                    }
                )

    summary = {
        "metadata": {
            "session_id": session_id,
            "total_duration_seconds": round(duration, 2),
            "total_images_interacted": len(rag_documents),
            "total_blinks": session_blinks,
        },
        "interactions": rag_documents,
    }

    json_path = f"results/{session_id}_RAG_report.json"
    full_json_path = os.path.join("static", json_path)
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    rag_result = generate_rag_answer_from_json(full_json_path)
    summary["rag_analysis"] = {
        "answer": rag_result["answer"],
        "source": rag_result["source"],
        "error": rag_result.get("error"),
        "questions_used": rag_result.get("questions_used", []),
    }

    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    heatmap_path = ""
    scatter_path = ""
    pdf_path = ""

    if _session_gaze_points:
        xs = [p[0] for p in _session_gaze_points]
        ys = [p[1] for p in _session_gaze_points]

        width, height = 800, 600
        heatmap = np.zeros((height, width))
        for x, y in _session_gaze_points:
            px, py = int(x * width), int(y * height)
            if 0 <= px < width and 0 <= py < height:
                heatmap[py, px] += 1

        heatmap = gaussian_filter(heatmap, sigma=35)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap, cmap="inferno", interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="Attention Intensity")
        ax.axvline(width / 2, color="white", linestyle="--", alpha=0.6)
        ax.axhline(height / 2, color="white", linestyle="--", alpha=0.6)
        ax.axis("off")
        ax.set_title(f"Gaze Heatmap - 2x2 Quadrants ({session_id})")
        heatmap_path = f"results/{session_id}_heatmap.png"
        fig.savefig(
            os.path.join("static", heatmap_path),
            bbox_inches="tight",
            facecolor="black",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, alpha=0.4, s=15, c="#00ff88", edgecolors="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.axvline(0.5, color="#00c8ff", linestyle="--", alpha=0.7)
        ax.axhline(0.5, color="#00c8ff", linestyle="--", alpha=0.7)
        ax.text(0.25, 0.25, "UP-LEFT", color="white", alpha=0.5, ha="center", va="center")
        ax.text(0.75, 0.25, "UP-RIGHT", color="white", alpha=0.5, ha="center", va="center")
        ax.text(0.25, 0.75, "DOWN-LEFT", color="white", alpha=0.5, ha="center", va="center")
        ax.text(0.75, 0.75, "DOWN-RIGHT", color="white", alpha=0.5, ha="center", va="center")
        ax.set_title(f"Gaze Scatter Plot ({session_id})", color="white")
        ax.set_facecolor("#080b10")
        fig.patch.set_facecolor("#080b10")
        ax.tick_params(colors="white")
        scatter_path = f"results/{session_id}_scatter.png"
        fig.savefig(
            os.path.join("static", scatter_path),
            bbox_inches="tight",
            facecolor="#080b10",
        )
        plt.close(fig)

    pdf_path = f"results/{session_id}_report.pdf"
    full_pdf_path = os.path.join("static", pdf_path)
    _build_session_pdf(
        full_pdf_path,
        summary,
        os.path.join("static", heatmap_path) if heatmap_path else "",
        os.path.join("static", scatter_path) if scatter_path else "",
    )

    _session_gaze_points = []

    return jsonify(
        {
            "summary": summary,
            "heatmap": heatmap_path,
            "scatter": scatter_path,
            "json_file": json_path,
            "pdf_file": pdf_path,
            "total_blinks": session_blinks,
            "rag_analysis": summary["rag_analysis"],
        }
    )


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    print("=" * 50)
    print("  CopyEmotion 2x2 - Flask Backend")
    print("=" * 50)
    print(f"Open in browser: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
