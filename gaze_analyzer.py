import math
import time
import cv2
import mediapipe as mp
from collections import deque, Counter

import config


class GazeAnalyzer:

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

        # Calibration
        self.neutral_h = None
        self.neutral_v = None
        self.calib_buffer = []
        self.is_calibrated = False

        # --- NEW: dynamic vertical range tracking ---
        # self.dv_min = None
        # self.dv_max = None
        # self.vertical_initialized = False

        # smoothing
        self.smooth_x = 0.5
        self.smooth_y = 0.5

        # region detection
        self.region_history = deque(maxlen=config.REGION_HISTORY_SIZE)
        self._candidate_region = None
        self._candidate_streak = 0
        self._last_emitted_region = None

        # blink detection
        self.blink_state = 0
        self.blink_counter = 0
        self.blink_debounce = 0

        # data
        self.gaze_points = []
        self.current_layout = 1

    # -------------------------------------------------
    # RESET FUNCTIONS
    # -------------------------------------------------

    def reset(self):
        self.gaze_points = []
        self.region_history.clear()
        self._candidate_region = None
        self._candidate_streak = 0
        self._last_emitted_region = None

        self.dv_min = None
        self.dv_max = None
        self.vertical_initialized = False

        self.blink_counter = 0

    def set_layout(self, num_images):
        if num_images in config.LAYOUTS:
            self.current_layout = num_images

    # -------------------------------------------------
    # HELPER FUNCTIONS
    # -------------------------------------------------

    def _blink_ratio(self, landmarks, eye_indices):
        t, b, l, r = [landmarks[i] for i in eye_indices]
        vertical = math.hypot(t.x - b.x, t.y - b.y)
        horizontal = math.hypot(l.x - r.x, l.y - r.y)
        return vertical / max(horizontal, 1e-6)

    def _eye_gaze_ratio(self, landmarks, iris_id, eye_indices):
        iris = landmarks[iris_id]
        t, b, l, r = [landmarks[i] for i in eye_indices]
        gx = (iris.x - l.x) / max(r.x - l.x, 1e-6)
        gy = (iris.y - t.y) / max(b.y - t.y, 1e-6)
        return gx, gy

    def _coords_to_region(self, x, y):
        layout = config.LAYOUTS[self.current_layout]
        cols = layout["cols"]
        rows = layout["rows"]

        # Column hysteresis
        if cols == 2:
            # Instead of a single line at 0.5, we have a buffer zone (0.45 to 0.55).
            left_threshold = 0.45
            right_threshold = 0.55

            if self._last_emitted_region is None:
                col_idx = 0 if x < 0.5 else 1
            else:
                prev_col = self._last_emitted_region % cols
                if prev_col == 0:
                    col_idx = 1 if x > right_threshold else 0
                else:
                    col_idx = 0 if x < left_threshold else 1
        else:
            col_idx = min(int(x * cols), cols - 1)

        # Row hysteresis
        if rows == 2:
            top_threshold = 0.45
            bottom_threshold = 0.55

            if self._last_emitted_region is None:
                row_idx = 0 if y < 0.5 else 1
            else:
                prev_row = self._last_emitted_region // cols
                if prev_row == 0:
                    row_idx = 1 if y > bottom_threshold else 0
                else:
                    row_idx = 0 if y < top_threshold else 1
        else:
            row_idx = 0

        return row_idx * cols + col_idx


    def _update_stable_gate(self, voted_region, confidence):

        if confidence < config.CONFIDENCE_THRESHOLD:
            self._candidate_streak = 0
            return False, self._last_emitted_region

        if voted_region == self._candidate_region:
            self._candidate_streak += 1
        else:
            self._candidate_region = voted_region
            self._candidate_streak = 1

        if (self._candidate_streak >= config.STABLE_FRAMES and
                voted_region != self._last_emitted_region):

            self._last_emitted_region = voted_region
            return True, voted_region

        return False, self._last_emitted_region

    # -------------------------------------------------
    # MAIN PROCESSING
    # -------------------------------------------------

    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        # --- Blink detection ---
        left_r = self._blink_ratio(lm, config.LEFT_EYE)
        right_r = self._blink_ratio(lm, config.RIGHT_EYE)
        avg_ratio = (left_r + right_r) / 2

        eyes_closed = avg_ratio < config.BLINK_THRESHOLD

        if eyes_closed and self.blink_state == 0 and self.blink_debounce == 0:
            self.blink_counter += 1
            self.blink_state = 1
            self.blink_debounce = config.BLINK_DEBOUNCE_FRAMES
        elif not eyes_closed:
            self.blink_state = 0

        if self.blink_debounce > 0:
            self.blink_debounce -= 1

        # ---------------------------------------------------------
        # NEW LOGIC: Stop Tracking When Eyes Are Closed
        # ---------------------------------------------------------
        if eyes_closed:
            # If eyes are closed, return the LAST known good region.
            # Do NOT update smoothing or voting history.
            return {
                "region": self._last_emitted_region, # Keep the old region
                "confidence": 0.0,                   # Zero confidence
                "blink_count": self.blink_counter,
                "eyes_closed": True,
                "region_changed": False              # Do not trigger a change
            }
        # ---------------------------------------------------------

        # --- Raw gaze (Only runs if eyes are OPEN) ---
        lh, lv = self._eye_gaze_ratio(lm, config.LEFT_IRIS, config.LEFT_EYE)
        rh, rv = self._eye_gaze_ratio(lm, config.RIGHT_IRIS, config.RIGHT_EYE)

        avg_h = (lh + rh) / 2
        avg_v = (lv + rv) / 2

        # --- Calibration ---
        if not self.is_calibrated:
            self.calib_buffer.append((avg_h, avg_v))

            if len(self.calib_buffer) >= config.CALIBRATION_FRAMES:
                self.neutral_h = sum(x[0] for x in self.calib_buffer) / len(self.calib_buffer)
                self.neutral_v = sum(x[1] for x in self.calib_buffer) / len(self.calib_buffer)
                self.is_calibrated = True
                print("✅ Calibration complete")

            return {
                "calibrating": True,
                "progress": len(self.calib_buffer) / config.CALIBRATION_FRAMES,
                "blink_count": self.blink_counter,
            }

        # --- Normal Tracking Logic ---
        dh = avg_h - self.neutral_h
        dv = avg_v - self.neutral_v

        if abs(dh) < config.DEAD_ZONE:
            dh = 0
        if abs(dv) < config.DEAD_ZONE:
            dv = 0

        # Horizontal
        raw_x = 0.5 - dh * config.GAIN_X
        raw_x = max(0.0, min(1.0, raw_x))

        # Vertical (Symmetric)
        VERTICAL_GAIN = 3.2
        raw_y = 0.5 + dv * VERTICAL_GAIN
        raw_y = max(0.0, min(1.0, raw_y))

        # Smoothing
        self.smooth_x += (raw_x - self.smooth_x) * config.SMOOTH_FACTOR
        self.smooth_y += (raw_y - self.smooth_y) * config.SMOOTH_FACTOR

        # Region voting
        instant_region = self._coords_to_region(self.smooth_x, self.smooth_y)
        self.region_history.append(instant_region)

        counts = Counter(self.region_history)
        voted_region, votes = counts.most_common(1)[0]
        confidence = votes / len(self.region_history)

        region_changed, stable_region = self._update_stable_gate(
            voted_region, confidence
        )

        self.gaze_points.append({
            "x": self.smooth_x,
            "y": self.smooth_y,
            "region": stable_region,
            "timestamp": time.time(),
        })

        return {
            "region": stable_region,
            "confidence": confidence,
            "blink_count": self.blink_counter,
            "eyes_closed": False,
            "region_changed": region_changed,
        }

    # -------------------------------------------------

    def get_analytics(self):
        return {}

    def get_heatmap_data(self):
        return [(p["x"], p["y"]) for p in self.gaze_points]