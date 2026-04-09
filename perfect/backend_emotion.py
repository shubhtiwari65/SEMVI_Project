"""
backend_emotion.py — Robust emotion detector using DeepFace
─────────────────────────────────────────────────────────────
Replaces FER (which silently fails on many systems) with DeepFace.

Install:  pip install deepface opencv-python
"""

import threading
import time
from collections import deque, Counter
import cv2

_DEEPFACE_OK = False
_INIT_ERROR  = ""
DeepFace     = None

try:
    from deepface import DeepFace as _DF
    DeepFace = _DF
    _DEEPFACE_OK = True
    print("[Emotion] DeepFace imported OK")
except ImportError:
    _INIT_ERROR = "DeepFace not installed. Run: pip install deepface"
    print(f"[Emotion] {_INIT_ERROR}")
except Exception as e:
    _INIT_ERROR = str(e)
    print(f"[Emotion] Import error: {e}")

SKIP_FRAMES    = 4
SMOOTH_VOTES   = 5
INFER_WIDTH    = 320
VALID_EMOTIONS = {"angry","disgust","fear","happy","sad","surprise","neutral"}


class EmotionDetector:
    def __init__(self):
        self._emotion    = "waiting"
        self._confidence = 0
        self._status     = _INIT_ERROR if not _DEEPFACE_OK else "ready"
        self._lock       = threading.Lock()
        self._running    = False
        self._frame      = None
        self._frame_id   = 0
        self._last_id    = -1
        self._frame_lock = threading.Lock()
        self._votes      = deque(maxlen=SMOOTH_VOTES)
        self._skip_ctr   = 0

    def push_frame(self, frame):
        with self._frame_lock:
            self._frame = frame.copy()
            self._frame_id += 1

    def get_emotion(self):
        with self._lock:
            return self._emotion, self._confidence

    def get_status(self):
        return self._status

    def start(self):
        if not _DEEPFACE_OK:
            print(f"[Emotion] Cannot start — {self._status}")
            return
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.name = "EmotionThread"
        t.start()
        print("[Emotion] Thread started")

    def stop(self):
        self._running = False

    def _loop(self):
        self._status = "running"
        print("[Emotion] Loop running...")
        errors = 0
        while self._running:
            with self._frame_lock:
                frame    = self._frame
                frame_id = self._frame_id

            if frame is None:
                time.sleep(0.05)
                continue

            if frame_id == self._last_id:
                time.sleep(0.02)
                continue

            self._skip_ctr += 1
            if self._skip_ctr < SKIP_FRAMES:
                self._last_id = frame_id
                time.sleep(0.01)
                continue
            self._skip_ctr = 0
            self._last_id  = frame_id

            try:
                emo, conf = self._infer(frame)
                errors = 0
            except Exception as e:
                errors += 1
                if errors <= 3 or errors % 20 == 0:
                    print(f"[Emotion] Error #{errors}: {e}")
                time.sleep(0.15)
                continue

            self._votes.append(emo)
            smoothed = Counter(self._votes).most_common(1)[0][0]
            with self._lock:
                self._emotion    = smoothed
                self._confidence = conf

        self._status = "stopped"

    def _infer(self, frame):
        h, w  = frame.shape[:2]
        scale = INFER_WIDTH / w
        small = cv2.resize(frame, (INFER_WIDTH, int(h * scale)))

        results = DeepFace.analyze(
            small,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )

        if not results:
            return "neutral", 0

        res  = results[0] if isinstance(results, list) else results
        emos = res.get("emotion", {})
        if not emos:
            return "neutral", 0

        top_emo  = res.get("dominant_emotion", max(emos, key=emos.get)).lower()
        top_conf = int(emos.get(top_emo, 0))

        if top_emo not in VALID_EMOTIONS:
            top_emo = "neutral"

        return top_emo, top_conf


emotion_detector = EmotionDetector()