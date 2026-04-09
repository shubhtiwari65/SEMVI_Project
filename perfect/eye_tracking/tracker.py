import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

class EyeTracker:
    # ── Pure Cross Calibration ──
    CAL_STEPS  = ['CENTER', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    CAL_SECS   = 2.0   

    def __init__(self, smoothing_frames=7): 
        print("Initializing MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.L_EYE_L = 33; self.L_EYE_R = 133
        self.R_EYE_L = 362; self.R_EYE_R = 263
        self.L_IRIS  = 468; self.R_IRIS  = 473
        self.L_EYE_TOP = 159; self.L_EYE_BOT = 145
        self.R_EYE_TOP = 386; self.R_EYE_BOT = 374

        # Standard rolling median buffer (No EMA)
        self.buf_x = deque(maxlen=smoothing_frames)
        self.buf_y = deque(maxlen=smoothing_frames)
        
        self.total_blinks = 0
        self._blink_state = False

        self.BLINK_T = 0.22 
        self.calibrated = False

        self._cal_active  = False
        self._cal_step    = 0
        self._cal_start   = 0.0
        self._cal_bufs_x  = [[] for _ in range(5)]
        self._cal_bufs_y  = [[] for _ in range(5)]
        self._cal_done    = False

        # Fallback calibration values
        self._c_x, self._c_y = 0.5, 0.5
        self._l_x, self._r_x = 0.4, 0.6
        self._u_y, self._d_y = 0.4, 0.6
        self.LEFT_T = 0.40; self.RIGHT_T = 0.60
        self.TOP_T  = 0.40; self.BOT_T   = 0.60

    def reset_calibration(self):
        self._cal_active = False
        self._cal_done = False
        self.calibrated = False
        self.buf_x.clear()
        self.buf_y.clear()

    def get_debug_info(self):
        return self.total_blinks, self.LEFT_T, self.RIGHT_T

    def _ear(self, lm, top, bot, left, right, w, h):
        v_dist = abs((lm[top].y * h) - (lm[bot].y * h))
        h_dist = abs((lm[left].x * w) - (lm[right].x * w))
        return v_dist / max(h_dist, 1e-6)

    def _ratio(self, lm, p1, p2, iris, axis, scale):
        val1 = getattr(lm[p1], axis) * scale
        val2 = getattr(lm[p2], axis) * scale
        val_i = getattr(lm[iris], axis) * scale
        span = abs(val2 - val1)
        if span < 2: return 0.5
        return float(np.clip(abs(val_i - val1) / span, 0.0, 1.0))

    def _get_raw_gaze(self, lm, w, h):
        # 1. Horizontal
        l_x = self._ratio(lm, self.L_EYE_L, self.L_EYE_R, self.L_IRIS, 'x', w)
        r_x = self._ratio(lm, self.R_EYE_L, self.R_EYE_R, self.R_IRIS, 'x', w)
        avg_x = (l_x + r_x) / 2.0

        # 2. Vertical: 3D Euclidean Distance (Robust against eyelid droop)
        iris_y = (lm[self.L_IRIS].y + lm[self.R_IRIS].y) / 2.0
        ref_y  = (lm[self.L_EYE_L].y + lm[self.R_EYE_R].y) / 2.0
        
        # We use the outer eye corners as an absolute 3D reference frame
        lx, ly, lz = lm[self.L_EYE_L].x, lm[self.L_EYE_L].y, lm[self.L_EYE_L].z
        rx, ry, rz = lm[self.R_EYE_R].x, lm[self.R_EYE_R].y, lm[self.R_EYE_R].z
        
        eye_span_3d = math.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
        if eye_span_3d < 1e-4: eye_span_3d = 0.1
        
        avg_y = (iris_y - ref_y) / eye_span_3d
        return avg_x, avg_y

    def start_calibration(self):
        self._cal_active = True
        self._cal_step   = 0
        self._cal_start  = time.time()
        self._cal_bufs_x = [[] for _ in range(5)]
        self._cal_bufs_y = [[] for _ in range(5)]
        self._cal_done   = False

    def calibration_status(self):
        if not self._cal_active and not self._cal_done:
            return {'active': False, 'done': False}
        if self._cal_done:
            return {'active': False, 'done': True}
        return {
            'active': True, 'done': False,
            'step': self._cal_step, 'step_name': self.CAL_STEPS[self._cal_step],
            'frac': round(min((time.time() - self._cal_start) / self.CAL_SECS, 1.0), 3)
        }

    def _finish_calibration(self):
        def med(buf): return float(np.median(buf)) if buf else 0.5
        
        self._c_x = med(self._cal_bufs_x[0]); self._c_y = med(self._cal_bufs_y[0])
        self._l_x = med(self._cal_bufs_x[1]); self._r_x = med(self._cal_bufs_x[2])
        self._u_y = med(self._cal_bufs_y[3]); self._d_y = med(self._cal_bufs_y[4])

        self.LEFT_T  = (self._c_x + self._l_x) / 2.0
        self.RIGHT_T = (self._c_x + self._r_x) / 2.0
        self.TOP_T   = (self._c_y + self._u_y) / 2.0
        self.BOT_T   = (self._c_y + self._d_y) / 2.0

        self.calibrated = True

    def process(self, frame):
        try:
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            if not res.multi_face_landmarks: return "NO_FACE"
            lm = res.multi_face_landmarks[0].landmark

            # 1. Blinks
            avg_ear = (self._ear(lm, self.L_EYE_TOP, self.L_EYE_BOT, self.L_EYE_L, self.L_EYE_R, w, h) + 
                       self._ear(lm, self.R_EYE_TOP, self.R_EYE_BOT, self.R_EYE_L, self.R_EYE_R, w, h)) / 2.0

            is_blink = False
            if avg_ear < self.BLINK_T:
                if not self._blink_state:
                    self.total_blinks += 1
                    self._blink_state = True
                is_blink = True
            else:
                self._blink_state = False

            # 2. Gaze coordinates
            raw_x, raw_y = self._get_raw_gaze(lm, w, h)

            if self._cal_active:
                self._cal_bufs_x[self._cal_step].append(raw_x)
                self._cal_bufs_y[self._cal_step].append(raw_y)
                if time.time() - self._cal_start >= self.CAL_SECS:
                    self._cal_step += 1
                    self._cal_start = time.time()
                    if self._cal_step >= len(self.CAL_STEPS):
                        self._cal_active = False
                        self._cal_done = True
                        self._finish_calibration()

            # Lightweight Smoothing Math (No EMA, just Rolling Median)
            self.buf_x.append(raw_x)
            self.buf_y.append(raw_y)
            med_x = float(np.median(self.buf_x))
            med_y = float(np.median(self.buf_y))

            if not self.calibrated: 
                return {"region": "CENTER", "x": med_x, "y": med_y, "blink": is_blink}

            # --- Explicit Quadrant Splitting ---
            # Determine horizontal quadrant relative to explicit center point
            if self._l_x < self._r_x:
                pos_x = "LEFT" if med_x < self._c_x else "RIGHT"
            else:
                pos_x = "LEFT" if med_x > self._c_x else "RIGHT"

            # Determine vertical quadrant relative to explicit center point
            if self._u_y < self._d_y:
                pos_y = "TOP" if med_y < self._c_y else "BOT"
            else:
                pos_y = "TOP" if med_y > self._c_y else "BOT"

            # Plot normalization 
            lo_x = min(self.LEFT_T, self.RIGHT_T); hi_x = max(self.LEFT_T, self.RIGHT_T)
            lo_y = min(self.TOP_T, self.BOT_T);    hi_y = max(self.TOP_T, self.BOT_T)

            norm_x = np.clip((med_x - lo_x) / (hi_x - lo_x + 1e-6), 0, 1)
            norm_y = np.clip((med_y - lo_y) / (hi_y - lo_y + 1e-6), 0, 1)

            if self._l_x > self._r_x: norm_x = 1.0 - norm_x
            if self._u_y > self._d_y: norm_y = 1.0 - norm_y

            return {
                "region": f"{pos_y}_{pos_x}",
                "x": norm_x,
                "y": norm_y,
                "blink": is_blink
            }

        except Exception as e:
            return "ERROR"