import os, threading, time, json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from flask import Flask, Response, jsonify, render_template, request
from eye_tracking.tracker import EyeTracker
from backend_emotion import emotion_detector

app = Flask(__name__, template_folder="templates", static_folder="static")

_gaze                = "NO_FACE"
_gaze_lock           = threading.Lock()
_cam_ready           = False
_frame_count         = 0
_cam_enabled         = False
_session_gaze_points = []
_session_start_time  = None

def get_gaze():
    with _gaze_lock: return _gaze

def set_gaze(val):
    global _gaze
    with _gaze_lock: _gaze = val

# ── Images ────────────────────────────────────────────────────────────────────
IMAGE_DIR = os.path.join("static", "images")
_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}

def _image_emotion(filename):
    n = filename.lower()
    if n.startswith("su"): return "surprise"
    if n.startswith("an") or n.startswith("a"): return "angry"
    if n.startswith("h"):  return "happy"
    if n.startswith("sa") or n.startswith("s"): return "sad"
    if n.startswith("d"):  return "disgust"
    if n.startswith("f"):  return "fear"
    if n.startswith("n"):  return "neutral"
    return "unknown"

def _load_images():
    if not os.path.isdir(IMAGE_DIR): 
        os.makedirs(IMAGE_DIR, exist_ok=True)
        return [], {}
    files = sorted(f for f in os.listdir(IMAGE_DIR) if os.path.splitext(f)[1].lower() in _EXTS)
    return files, {f: _image_emotion(f) for f in files}

IMAGE_LIST, IMAGE_EMOTIONS = _load_images()
print(f"[Images] {len(IMAGE_LIST)} images loaded")

# ── Tracker ───────────────────────────────────────────────────────────────────
tracker = EyeTracker(smoothing_frames=7)

# ── Camera thread ─────────────────────────────────────────────────────────────
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_detector.start()
    _cam_ready = True
    print("[Camera] Ready — waiting for session to start")

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
            if res["blink"]:
                set_gaze("BLINK")
            else:
                set_gaze(res["region"])
                # Track normalized points for the heatmap
                if tracker.calibrated: 
                    _session_gaze_points.append((res["x"], res["y"]))
        else:
            set_gaze(res)

        emotion_detector.push_frame(frame)

threading.Thread(target=_camera_thread, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
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
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

@app.route("/images")
def list_images():
    return jsonify(images=IMAGE_LIST, emotions=IMAGE_EMOTIONS)

@app.route("/emotion")
def emotion():
    emo, conf = emotion_detector.get_emotion()
    return jsonify(emotion=emo, confidence=conf)

@app.route("/blinks")
def get_blinks():
    return jsonify(blinks=tracker.total_blinks)

# ── Calibration ───────────────────────────────────────────────────────────────
@app.route("/calibrate", methods=["POST"])
def calibrate():
    tracker.reset_calibration()
    tracker.start_calibration()
    return jsonify(ok=True)

@app.route("/calibrate/status")
def calibrate_status():
    return jsonify(tracker.calibration_status())

# ── Camera on/off ─────────────────────────────────────────────────────────────
@app.route("/camera/start", methods=["POST"])
def camera_start():
    global _cam_enabled, _session_start_time
    _cam_enabled = True
    _session_start_time = time.time()
    print("[Camera] Enabled — session started")
    return jsonify(ok=True)

@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    global _cam_enabled
    _cam_enabled = False
    set_gaze("NO_FACE")
    print("[Camera] Disabled — session ended")
    return jsonify(ok=True)

# ── Generate Report & Plots ───────────────────────────────────────────────────
@app.route("/api/stop_session", methods=["POST"])
def stop_session():
    global _session_start_time, _session_gaze_points
    
    frontend_data = request.json
    duration = time.time() - _session_start_time if _session_start_time else 0
    session_id = f"session_{int(time.time())}"
    os.makedirs("static/results", exist_ok=True)

    formatted_images = []
    if frontend_data:
        for img_name, data in frontend_data.items():
            if not img_name.startswith("TOP_") and not img_name.startswith("BOT_"):
                formatted_images.append({
                    "name": img_name,
                    "time_spent": data.get("timeMs", 0) / 1000.0,
                    "dominant_emotion": data.get("imgEmo", "neutral"),
                    "fixation_count": 0 
                })

    summary = {
        "session_id": session_id,
        "duration": round(duration, 2),
        "total_blinks": tracker.total_blinks,
        "num_images": len(formatted_images),
        "pages": len(formatted_images) // 4,
        "images": formatted_images
    }

    heatmap_path = ""
    scatter_path = ""

    if _session_gaze_points:
        xs = [p[0] for p in _session_gaze_points]
        ys = [p[1] for p in _session_gaze_points]
        
        width, height = 800, 600
        heatmap = np.zeros((height, width))
        for x, y in _session_gaze_points:
            px, py = int(x * width), int(y * height)
            if 0 <= px < width and 0 <= py < height:
                heatmap[py, px] += 1
        heatmap = gaussian_filter(heatmap, sigma=40)
        if heatmap.max() > 0: heatmap /= heatmap.max()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(heatmap, cmap="hot", interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="Attention Intensity")
        ax.axis("off")
        ax.set_title(f"Gaze Heatmap - {session_id}")
        heatmap_path = f"results/{session_id}_heatmap.png"
        fig.savefig(f"static/{heatmap_path}", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xs, ys, alpha=0.3, s=10, c="royalblue")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"Gaze Scatter - {session_id}")
        ax.set_xlabel("X (normalised)")
        ax.set_ylabel("Y (normalised)")
        scatter_path = f"results/{session_id}_scatter.png"
        fig.savefig(f"static/{scatter_path}", bbox_inches="tight")
        plt.close(fig)

    _session_gaze_points = []
    
    return jsonify({
        "summary": summary,
        "heatmap": heatmap_path,
        "scatter": scatter_path
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  CopyEmotion 2x2 — Flask Backend")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)