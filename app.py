"""
Main Flask Application for Eye Tracking + Emotion Analysis
===========================================================
Changes from original app.py
-----------------------------
* Imports EmotionAnalyzer (emotion_analyzer.py)
* tracking_loop passes each frame to emotion_analyzer.analyze()
* gaze_update socket event now carries real-time emotion + scores
* /api/stop_session response includes per-region emotion summary
* /api/recalibrate also resets emotion history
"""

import os
import cv2
import time
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from datetime import datetime

import config
from gaze_analyzer import GazeAnalyzer
from data_manager  import DataManager
from emotion_analyzer import EmotionAnalyzer          # ← NEW

# ──────────────────────────────────────────────────────────────────────────────
# Environment flags  (must be set before importing mediapipe / tensorflow)
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"]       = "2"

# ──────────────────────────────────────────────────────────────────────────────
# Flask / SocketIO
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "eye-tracking-secret-2025"
socketio = SocketIO(app, async_mode="threading",
                    cors_allowed_origins=config.CORS_ALLOWED_ORIGINS)

# ──────────────────────────────────────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────────────────────────────────────
gaze_analyzer    = GazeAnalyzer()
data_manager     = DataManager()
emotion_analyzer = EmotionAnalyzer()               # ← NEW  (shares no camera)

tracking_active    = False
current_session_id = None


# ──────────────────────────────────────────────────────────────────────────────
# Tracking thread  (single camera owner — feeds both gaze AND emotion)
# ──────────────────────────────────────────────────────────────────────────────
def tracking_loop():
    """
    Background thread: reads webcam frames, runs gaze analysis, runs emotion
    analysis on the same frame, then emits a combined SocketIO event.

    Design notes
    ------------
    * Only ONE VideoCapture is created here.  EmotionAnalyzer receives a copy
      of the frame via its non-blocking .analyze() call so there is never a
      second capture device opened.
    * Emotion analysis runs in its OWN internal thread (see EmotionAnalyzer)
      so it never blocks the ~100 Hz gaze loop.
    """
    global tracking_active

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    print(f"📷 Camera initialised (index {config.CAMERA_INDEX})")

    while True:
        if not tracking_active:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        # ── Gaze analysis ────────────────────────────────────────────
        gaze_data = gaze_analyzer.process_frame(frame)

        if gaze_data is None:
            socketio.emit("status", {"message": "No face detected"})
            time.sleep(0.03)
            continue

        # ── calibration progress ──────────────────────────────────────
        if gaze_data.get("calibrating"):
            socketio.emit("calibration", {
                "progress":    gaze_data["progress"],
                "blink_count": gaze_data["blink_count"],
            })
            time.sleep(0.03)
            continue

        # ── Emotion analysis (non-blocking) ───────────────────────────
        current_region = gaze_data.get("region")   # may be None during warmup

        emotion_data = emotion_analyzer.analyze(frame, current_region)   # ← NEW
        # emotion_data = {
        #   "emotion": "happy",        <- smoothed dominant for this region
        #   "scores":  {label: float}, <- raw DeepFace probabilities
        #   "region":  int | None,
        # }

        # ── live session update ───────────────────────────────────────
        if current_session_id:
            data_manager.update_session(gaze_data)

        # ── emit combined event to frontend ──────────────────────────
        socketio.emit("gaze_update", {
            # ── gaze ──
            "region":         current_region,
            "confidence":     gaze_data.get("confidence", 0.0),
            "blink_count":    gaze_data.get("blink_count", 0),
            "eyes_closed":    gaze_data.get("eyes_closed", False),
            "region_changed": gaze_data.get("region_changed", False),

            # ── emotion (live, per region) ──────────────────────────
            "emotion":        emotion_data["emotion"],          # ← NEW
            "emotion_scores": emotion_data["scores"],           # ← NEW
            "emotion_region": emotion_data["region"],           # ← NEW
        })

        time.sleep(0.01)   # ≈ 100 Hz max

    cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/images", methods=["GET"])
def list_images():
    """Return sorted list of image filenames in static/images/."""
    image_dir = Path(config.IMAGE_FOLDER)
    image_dir.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    images = sorted(
        f.name for f in image_dir.iterdir()
        if f.suffix.lower() in exts
    )
    return jsonify({"images": images})


@app.route("/api/start_session", methods=["POST"])
def start_session():
    """Start a brand-new tracking + emotion session."""
    global tracking_active, current_session_id

    data        = request.json or {}
    num_images  = data.get("num_images",  1)
    image_names = data.get("image_names", [])

    session_id         = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    current_session_id = session_id

    gaze_analyzer.reset()
    gaze_analyzer.set_layout(num_images)
    emotion_analyzer.reset()                        # ← NEW: clear emotion history

    data_manager.start_session(session_id, num_images, image_names)

    tracking_active = True

    return jsonify({
        "success":    True,
        "session_id": session_id,
        "message":    "Session started – look at the centre to calibrate.",
    })


@app.route("/api/navigate", methods=["POST"])
def navigate():
    """
    Called when the user presses Next / Previous.
    Saves analytics for the current image batch, resets region gate
    and emotion history, registers the new image set.
    """
    global current_session_id

    if not current_session_id:
        return jsonify({"success": False, "message": "No active session"}), 400

    data        = request.json or {}
    image_names = data.get("image_names", [])
    page_index  = data.get("page_index",  0)

    # Snapshot gaze analytics for the outgoing batch
    analytics = gaze_analyzer.get_analytics()
    data_manager.save_page_analytics(current_session_id, page_index, analytics)

    # ── Snapshot per-region emotion for the outgoing batch ───────────
    # Attach emotion summary to data_manager so it surfaces in the final report
    emotion_summary = emotion_analyzer.get_region_emotion_summary()   # ← NEW
    data_manager.save_page_emotion(                                    # ← NEW
        current_session_id, page_index, emotion_summary
    )

    # Reset gaze region gate for new images
    gaze_analyzer.reset_image_set()

    # Reset emotion history for new images                             # ← NEW
    emotion_analyzer.reset()

    # Register the new image set in data-manager
    data_manager.switch_image_set(current_session_id, image_names)

    return jsonify({"success": True, "page_index": page_index})


@app.route("/api/stop_session", methods=["POST"])
def stop_session():
    """Stop tracking + emotion, generate visualisations, return summary."""
    global tracking_active, current_session_id

    if not current_session_id:
        return jsonify({"success": False, "message": "No active session"}), 400

    tracking_active = False

    analytics = gaze_analyzer.get_analytics()
    session_id = data_manager.end_session(current_session_id, analytics)

    # ── Emotion summary for the final page ───────────────────────────
    emotion_summary = emotion_analyzer.get_region_emotion_summary()   # ← NEW
    data_manager.save_page_emotion(                                    # ← NEW
        session_id,
        data_manager.get_current_page_index(session_id),
        emotion_summary,
    )

    gaze_xy      = gaze_analyzer.get_heatmap_data()
    num_images   = data_manager.sessions.get(session_id, {}).get("num_images", 1)
    heatmap_path = data_manager.generate_heatmap(session_id, gaze_xy)
    scatter_path = data_manager.generate_scatter_plot(session_id, gaze_xy, num_images)
    summary      = data_manager.get_session_summary(session_id)

    current_session_id = None

    return jsonify({
        "success":    True,
        "session_id": session_id,
        "summary":    summary,
        "heatmap":    heatmap_path.replace("static/", "") if heatmap_path else None,
        "scatter":    scatter_path.replace("static/", "") if scatter_path else None,
        # ── per-region emotion summary returned to frontend ──────────
        "emotion_summary": emotion_summary,                            # ← NEW
    })


@app.route("/api/recalibrate", methods=["POST"])
def recalibrate():
    """Trigger a fresh calibration without stopping the session."""
    gaze_analyzer.is_calibrated = False
    gaze_analyzer.calib_buffer  = []
    gaze_analyzer.neutral_h     = None
    gaze_analyzer.neutral_v     = None
    emotion_analyzer.reset()                        # ← NEW: also reset emotion
    return jsonify({"success": True, "message": "Recalibration started"})


@app.route("/api/session/<session_id>", methods=["GET"])
def get_session_data(session_id):
    summary = data_manager.get_session_summary(session_id)
    if summary is None:
        return jsonify({"success": False, "message": "Session not found"}), 404
    return jsonify({"success": True, "data": summary})


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=tracking_loop, daemon=True).start()

    print("=" * 52)
    print("👁️  Eye Tracking + Emotion Analysis System")
    print("=" * 52)
    print(f"   Server  : http://localhost:{config.PORT}")
    print(f"   Images  : {config.IMAGE_FOLDER}/")
    print(f"   Results : {config.RESULTS_FOLDER}/")
    print("=" * 52)

    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        use_reloader=False,
    )
