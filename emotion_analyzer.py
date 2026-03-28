"""
emotion_analyzer.py
--------------------
Wraps DeepFace emotion detection and ties it to the currently active
gaze region.  Designed to be called from the SAME tracking thread that
already holds the webcam frame, so there is no second VideoCapture.

Usage (inside tracking_loop in app.py):
    emotion_analyzer = EmotionAnalyzer()
    ...
    frame_emotion = emotion_analyzer.analyze(frame, current_region)
    # -> {"emotion": "happy", "scores": {...}, "region": 1}
"""

from deepface import DeepFace
from collections import defaultdict, deque
import threading
import time


# ---------------------------------------------------------------------------
# Per-region emotion history window (smoothing / majority-vote)
# ---------------------------------------------------------------------------
_HISTORY_SIZE = 10          # rolling window per region
_ANALYSIS_INTERVAL = 0.35   # seconds between DeepFace calls (it is slow)


class EmotionAnalyzer:
    """
    Thread-safe, non-blocking emotion analyser.

    Because DeepFace.analyze() is CPU-heavy (~150-400 ms), we run it in a
    background thread and return the latest cached result immediately so the
    main tracking loop is never blocked.
    """

    def __init__(self):
        # Latest result from the background thread
        self._lock = threading.Lock()
        self._latest_emotion: str = "Detecting..."
        self._latest_scores: dict = {}
        self._latest_region: int | None = None

        # Per-region history for smoothed dominant emotion
        self._region_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=_HISTORY_SIZE)
        )
        # Per-region accumulated emotion scores across the whole session
        # { region_id -> { emotion_label -> total_score } }
        self._region_scores: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._region_sample_count: dict[int, int] = defaultdict(int)

        # Background worker state
        self._pending_frame = None
        self._pending_region: int | None = None
        self._last_analysis_time: float = 0.0
        self._worker_busy = False

        # Kick off the worker thread once
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame, current_region: int | None):
        """
        Non-blocking call.  Queues a new frame for analysis if the worker
        is free and enough time has elapsed; returns the last cached result.

        Parameters
        ----------
        frame          : numpy BGR frame (same frame used for gaze analysis)
        current_region : integer region index from GazeAnalyzer (may be None)

        Returns
        -------
        dict with keys:
            emotion        – dominant emotion label (smoothed per region)
            scores         – raw probability dict from DeepFace
            region         – region this emotion was captured for
        """
        now = time.time()

        # Queue a frame for the background thread if idle and interval elapsed
        if (not self._worker_busy and
                now - self._last_analysis_time >= _ANALYSIS_INTERVAL):
            with self._lock:
                self._pending_frame  = frame.copy()
                self._pending_region = current_region
            self._worker_busy = True      # set before thread picks it up

        with self._lock:
            emotion = self._latest_emotion
            scores  = dict(self._latest_scores)
            region  = self._latest_region

        # Smooth the emotion per region using history
        if region is not None:
            smoothed = self._smoothed_emotion(region)
            emotion  = smoothed if smoothed else emotion

        return {
            "emotion": emotion,
            "scores":  scores,
            "region":  region,
        }

    def get_region_emotion_summary(self) -> dict[int, dict]:
        """
        Return per-region averaged emotion scores and dominant emotion.
        Called by app.py when building the session stop payload.

        Returns
        -------
        {
          region_id (int): {
            "dominant_emotion": str,
            "avg_scores":       { label: float (0-100) },
            "sample_count":     int,
          },
          ...
        }
        """
        summary = {}
        with self._lock:
            scores_copy = {k: dict(v) for k, v in self._region_scores.items()}
            counts_copy = dict(self._region_sample_count)

        for rid, scores in scores_copy.items():
            n = counts_copy.get(rid, 1) or 1
            avg = {label: val / n for label, val in scores.items()}
            dominant = max(avg, key=avg.get) if avg else "unknown"
            summary[rid] = {
                "dominant_emotion": dominant,
                "avg_scores":       avg,
                "sample_count":     n,
            }
        return summary

    def get_region_dominant_emotion(self, region_id: int) -> str:
        """Quick lookup: dominant emotion for one region."""
        summary = self.get_region_emotion_summary()
        return summary.get(region_id, {}).get("dominant_emotion", "unknown")

    def reset(self):
        """Clear all history (called at session start)."""
        with self._lock:
            self._region_history.clear()
            self._region_scores.clear()
            self._region_sample_count.clear()
            self._latest_emotion  = "Detecting..."
            self._latest_scores   = {}
            self._latest_region   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Background thread: picks up queued frames and calls DeepFace."""
        while True:
            time.sleep(0.05)

            with self._lock:
                frame  = self._pending_frame
                region = self._pending_region

            if frame is None or not self._worker_busy:
                continue

            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                dominant = results[0]["dominant_emotion"]
                scores   = results[0]["emotion"]          # dict label->float

                with self._lock:
                    self._latest_emotion = dominant
                    self._latest_scores  = scores
                    self._latest_region  = region
                    self._pending_frame  = None

                    # Accumulate per-region
                    if region is not None:
                        self._region_history[region].append(dominant)
                        for label, val in scores.items():
                            self._region_scores[region][label] += val
                        self._region_sample_count[region] += 1

                self._last_analysis_time = time.time()

            except Exception:
                with self._lock:
                    self._latest_emotion = "No face"
                    self._pending_frame  = None
                self._last_analysis_time = time.time()

            finally:
                self._worker_busy = False

    def _smoothed_emotion(self, region: int) -> str | None:
        """Majority-vote over the rolling window for this region."""
        with self._lock:
            history = list(self._region_history.get(region, []))
        if not history:
            return None
        return max(set(history), key=history.count)
