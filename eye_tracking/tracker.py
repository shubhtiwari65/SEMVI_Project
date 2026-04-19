import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


class QuadrantEyeTracker:
    """
    4-quadrant gaze tracker: UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT.
    Calibration: CENTER -> UP -> DOWN -> LEFT -> RIGHT (5 steps).
    """

    CAL_STEPS = ["CENTER", "UP", "DOWN", "LEFT", "RIGHT"]
    CAL_SECS = 3.0
    CAL_THRESHOLD = 0.35

    _SF = 7
    _EWM_ALPHA = 0.40
    _WARMUP_FRAMES = 8
    _HYST = 0.15
    _BLINK_THRESHOLD = 0.18
    _BLINK_MIN_FRAMES = 2

    def __init__(self, smoothing_frames=None):
        print("[Quad] Initializing MediaPipe...")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[Quad] MediaPipe initialized")

        self.L_IRIS = 468
        self.R_IRIS = 473
        self.L_OUTER = 33
        self.L_INNER = 133
        self.R_INNER = 362
        self.R_OUTER = 263
        self.NOSE_TIP = 1
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454
        self.L_UPPER_LID = 159
        self.L_LOWER_LID = 145
        self.R_UPPER_LID = 386
        self.R_LOWER_LID = 374

        sf = smoothing_frames or self._SF
        self.buf_ud = deque(maxlen=sf)
        self.buf_lr = deque(maxlen=sf)
        self._ewm_ud = None
        self._ewm_lr = None
        self._frame_count = 0

        self.UP_T = -0.06
        self.DOWN_T = 0.06
        self.LEFT_T = 0.38
        self.RIGHT_T = 0.62

        self._center_baseline_ud = 0.0
        self._left_is_lower = True
        self.calibrated = False

        self._last_v = "CENTER"
        self._last_h = "CENTER"
        self._last_region = "CENTER"
        self._last_plot_x = 0.5
        self._last_plot_y = 0.5

        self._cal_active = False
        self._cal_step = 0
        self._cal_start = 0.0
        self._cal_bufs = [[] for _ in range(5)]
        self._cal_done = False
        self._cal_pending = False

        self.total_blinks = 0
        self._blink_frames = 0
        self._blink_active = False

    # Raw signals

    def _raw_vertical(self, lm, fw, fh):
        lox = lm[self.L_OUTER].x * fw
        loy = lm[self.L_OUTER].y * fh
        lix = lm[self.L_INNER].x * fw
        liy = lm[self.L_INNER].y * fh
        l_corner_y = (loy + liy) / 2.0
        l_eye_w = abs(lix - lox)
        l_iris_y = lm[self.L_IRIS].y * fh

        rox = lm[self.R_OUTER].x * fw
        roy = lm[self.R_OUTER].y * fh
        rix = lm[self.R_INNER].x * fw
        riy = lm[self.R_INNER].y * fh
        r_corner_y = (roy + riy) / 2.0
        r_eye_w = abs(rix - rox)
        r_iris_y = lm[self.R_IRIS].y * fh

        vals = []
        if l_eye_w > 3:
            vals.append((l_iris_y - l_corner_y) / l_eye_w)
        if r_eye_w > 3:
            vals.append((r_iris_y - r_corner_y) / r_eye_w)
        return float(np.mean(vals)) if vals else 0.0

    def _ratio_lr(self, lm, el, er, iris, fw):
        x_l = lm[el].x * fw
        x_r = lm[er].x * fw
        x_i = lm[iris].x * fw
        span = x_r - x_l
        if span < 2:
            return 0.5
        return float(np.clip((x_i - x_l) / span, 0.0, 1.0))

    def _head_yaw_offset(self, lm, fw):
        nose = lm[self.NOSE_TIP].x * fw
        left_c = lm[self.LEFT_CHEEK].x * fw
        right_c = lm[self.RIGHT_CHEEK].x * fw
        face_w = right_c - left_c
        if face_w < 10:
            return 0.0
        centre = (left_c + right_c) / 2.0
        yaw = (nose - centre) / face_w
        return float(np.clip(yaw * 0.30, -0.2, 0.2))

    def _raw_horizontal(self, lm, fw):
        l_r = self._ratio_lr(lm, self.L_OUTER, self.L_INNER, self.L_IRIS, fw)
        r_r = self._ratio_lr(lm, self.R_INNER, self.R_OUTER, self.R_IRIS, fw)
        avg = (l_r + r_r) / 2.0
        yaw = self._head_yaw_offset(lm, fw)
        return float(np.clip(avg + yaw, 0.0, 1.0))

    def _blink_ratio(self, lm, top_idx, bottom_idx, left_idx, right_idx, fw, fh):
        top_y = lm[top_idx].y * fh
        bottom_y = lm[bottom_idx].y * fh
        left_x = lm[left_idx].x * fw
        right_x = lm[right_idx].x * fw
        eye_width = abs(right_x - left_x)
        if eye_width < 3:
            return 1.0
        return abs(bottom_y - top_y) / eye_width

    def _eye_open_ratio(self, lm, fw, fh):
        left_ratio = self._blink_ratio(
            lm, self.L_UPPER_LID, self.L_LOWER_LID, self.L_OUTER, self.L_INNER, fw, fh
        )
        right_ratio = self._blink_ratio(
            lm, self.R_UPPER_LID, self.R_LOWER_LID, self.R_INNER, self.R_OUTER, fw, fh
        )
        return (left_ratio + right_ratio) / 2.0

    def _detect_blink(self, eye_open_ratio):
        if eye_open_ratio < self._BLINK_THRESHOLD:
            self._blink_frames += 1
            if self._blink_frames >= self._BLINK_MIN_FRAMES and not self._blink_active:
                self._blink_active = True
                self.total_blinks += 1
                return True
        else:
            self._blink_frames = 0
            self._blink_active = False
        return False

    # Smoothing

    def _flush_smooth(self):
        self.buf_ud.clear()
        self._ewm_ud = None
        self.buf_lr.clear()
        self._ewm_lr = None
        self._frame_count = 0

    def _smooth_ud(self, v):
        self._ewm_ud = (
            v
            if self._ewm_ud is None
            else self._EWM_ALPHA * v + (1 - self._EWM_ALPHA) * self._ewm_ud
        )
        self.buf_ud.append(self._ewm_ud)
        return float(np.median(self.buf_ud))

    def _smooth_lr(self, v):
        self._ewm_lr = (
            v
            if self._ewm_lr is None
            else self._EWM_ALPHA * v + (1 - self._EWM_ALPHA) * self._ewm_lr
        )
        self.buf_lr.append(self._ewm_lr)
        return float(np.median(self.buf_lr))

    # Classifiers

    def _classify_vertical(self, sv):
        up_enter = self.UP_T
        up_exit = self.UP_T * (1.0 - self._HYST)
        down_enter = self.DOWN_T
        down_exit = self.DOWN_T * (1.0 - self._HYST)

        if self._last_v == "CENTER":
            if sv < up_enter:
                self._last_v = "UP"
            elif sv > down_enter:
                self._last_v = "DOWN"
        elif self._last_v == "UP":
            if sv > up_exit:
                self._last_v = "CENTER"
        elif self._last_v == "DOWN":
            if sv < down_exit:
                self._last_v = "CENTER"

        return self._last_v

    def _classify_horizontal(self, sv):
        lo = min(self.LEFT_T, self.RIGHT_T)
        hi = max(self.LEFT_T, self.RIGHT_T)
        span = hi - lo

        left_enter = lo
        right_enter = hi
        left_exit = lo + span * self._HYST
        right_exit = hi - span * self._HYST

        if self._last_h == "CENTER":
            if sv < left_enter:
                self._last_h = "LEFT" if self._left_is_lower else "RIGHT"
            elif sv > right_enter:
                self._last_h = "RIGHT" if self._left_is_lower else "LEFT"
        elif self._last_h == "LEFT":
            exit_val = left_exit if self._left_is_lower else right_exit
            if sv > exit_val:
                self._last_h = "CENTER"
        elif self._last_h == "RIGHT":
            exit_val = right_exit if self._left_is_lower else left_exit
            if sv < exit_val:
                self._last_h = "CENTER"

        return self._last_h

    def _reset_hysteresis(self):
        self._last_v = "CENTER"
        self._last_h = "CENTER"

    def _combine(self, v_dir, h_dir):
        if v_dir == "CENTER" and h_dir == "CENTER":
            return "CENTER"
        if v_dir == "UP":
            if h_dir == "LEFT":
                return "UP-LEFT"
            if h_dir == "RIGHT":
                return "UP-RIGHT"
            return "UP"
        if v_dir == "DOWN":
            if h_dir == "LEFT":
                return "DOWN-LEFT"
            if h_dir == "RIGHT":
                return "DOWN-RIGHT"
            return "DOWN"
        return h_dir

    def _normalized_y(self, sv_ud):
        up_bound = self.UP_T * 2 if self.UP_T != 0 else -0.1
        dn_bound = self.DOWN_T * 2 if self.DOWN_T != 0 else 0.1
        denom = dn_bound - up_bound
        if abs(denom) < 1e-6:
            return 0.5
        return float(np.clip((sv_ud - up_bound) / denom, 0.0, 1.0))

    def _stable_result(self, blink=False):
        return {
            "region": self._last_region,
            "x": float(self._last_plot_x),
            "y": float(self._last_plot_y),
            "blink": blink,
        }

    # Calibration

    def start_calibration(self):
        self._cal_active = False
        self._cal_step = 0
        self._cal_start = time.time()
        self._cal_bufs = [[] for _ in range(5)]
        self._cal_done = False
        self._cal_pending = True   # signal that calibration was requested but not yet active
        self._flush_smooth()
        self._reset_hysteresis()
        self._cal_active = True
        self._cal_pending = False
        print(f"[Quad Cal] Step 0: Look {self.CAL_STEPS[0]}")

    def calibration_status(self):
        if getattr(self, '_cal_pending', False):
            # Calibration requested but camera thread hasn't picked it up yet
            return {"active": True, "done": False, "step": 0,
                    "step_name": self.CAL_STEPS[0], "step_count": len(self.CAL_STEPS),
                    "frac": 0.0, "calibrated": self.calibrated}
        if not self._cal_active and not self._cal_done:
            return {"active": False, "done": False, "calibrated": self.calibrated}
        if self._cal_done:
            return {"active": False, "done": True, "calibrated": self.calibrated}
        elapsed = time.time() - self._cal_start
        return {
            "active": True,
            "done": False,
            "step": self._cal_step,
            "step_name": self.CAL_STEPS[self._cal_step],
            "step_count": len(self.CAL_STEPS),
            "frac": round(min(elapsed / self.CAL_SECS, 1.0), 3),
            "calibrated": self.calibrated,
        }

    def _finish_calibration(self):
        def iqr_med(lst):
            if len(lst) < 4:
                return float(np.median(lst)) if lst else None
            arr = np.array(lst, dtype=float)
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            clean = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
            return float(np.median(clean)) if len(clean) else float(np.median(arr))

        def iqr_std(lst):
            if len(lst) < 4:
                return 0.01
            arr = np.array(lst, dtype=float)
            q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            iqr = q3 - q1
            clean = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
            return float(np.std(clean)) if len(clean) > 1 else float(np.std(arr))

        ud_c = iqr_med([v[0] for v in self._cal_bufs[0]])
        ud_u = iqr_med([v[0] for v in self._cal_bufs[1]])
        ud_d = iqr_med([v[0] for v in self._cal_bufs[2]])
        ud_std = iqr_std([v[0] for v in self._cal_bufs[0]])
        lr_c = iqr_med([v[1] for v in self._cal_bufs[0]])
        lr_l = iqr_med([v[1] for v in self._cal_bufs[3]])
        lr_r = iqr_med([v[1] for v in self._cal_bufs[4]])

        if None in [ud_c, ud_u, ud_d, lr_c, lr_l, lr_r]:
            print("[Quad Cal] Not enough samples - aborting")
            return

        print(
            f"[Quad Cal] UD  CENTER:{ud_c:.4f}  UP:{ud_u:.4f}  "
            f"DOWN:{ud_d:.4f}  sigma={ud_std:.4f}"
        )
        print(f"[Quad Cal] LR  CENTER:{lr_c:.4f}  LEFT:{lr_l:.4f}  RIGHT:{lr_r:.4f}")

        self._center_baseline_ud = ud_c
        u_rel = ud_u - ud_c
        d_rel = ud_d - ud_c
        if u_rel > d_rel:
            print("[Quad Cal] UD axis inverted - swapping")
            u_rel, d_rel = d_rel, u_rel

        noise_floor = max(ud_std * 2.5, 0.005)
        up_dz = max(abs(u_rel) * self.CAL_THRESHOLD, noise_floor)
        down_dz = max(abs(d_rel) * self.CAL_THRESHOLD, noise_floor)
        self.UP_T = round(max(-up_dz, u_rel * 0.5), 5)
        self.DOWN_T = round(min(down_dz, d_rel * 0.5), 5)

        if abs(u_rel) < 0.01:
            print("[Quad Cal] WARNING: UP range tiny - look higher during calibration")
        if abs(d_rel) < 0.01:
            print("[Quad Cal] WARNING: DOWN range tiny - look lower during calibration")

        lr_gap_l = abs(lr_c - lr_l)
        lr_gap_r = abs(lr_r - lr_c)
        self.LEFT_T = round(lr_c - lr_gap_l * (1.0 - self.CAL_THRESHOLD), 4)
        self.RIGHT_T = round(lr_c + lr_gap_r * (1.0 - self.CAL_THRESHOLD), 4)
        self._left_is_lower = lr_l < lr_r

        print(f"[Quad Cal] UD thresholds  UP_T:{self.UP_T:.4f}  DOWN_T:{self.DOWN_T:.4f}")
        print(
            f"[Quad Cal] LR thresholds  LEFT_T:{self.LEFT_T:.4f}  "
            f"RIGHT_T:{self.RIGHT_T:.4f}  left_is_lower:{self._left_is_lower}"
        )

        self.calibrated = True
        self._flush_smooth()
        self._reset_hysteresis()
        print("[Quad Cal] Done")

    def reset_calibration(self):
        self._cal_active = False
        self._cal_done = False
        self._cal_bufs = [[] for _ in range(5)]
        self._center_baseline_ud = 0.0
        self._left_is_lower = True
        self._flush_smooth()
        self._reset_hysteresis()
        self.calibrated = False
        self.UP_T = -0.06
        self.DOWN_T = 0.06
        self.LEFT_T = 0.38
        self.RIGHT_T = 0.62
        print("[Quad Cal] Reset")

    # Main process

    def process(self, frame):
        try:
            fh, fw = frame.shape[:2]
            res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                self._frame_count = 0
                self._blink_frames = 0
                self._blink_active = False
                return "NO_FACE"

            lm = res.multi_face_landmarks[0].landmark
            eye_open_ratio = self._eye_open_ratio(lm, fw, fh)
            blink = self._detect_blink(eye_open_ratio)
            raw_ud = self._raw_vertical(lm, fw, fh)
            raw_lr = self._raw_horizontal(lm, fw)

            if self._cal_active:
                self._cal_bufs[self._cal_step].append((raw_ud, raw_lr))
                elapsed = time.time() - self._cal_start
                if elapsed >= self.CAL_SECS:
                    next_step = self._cal_step + 1
                    if next_step >= len(self.CAL_STEPS):
                        self._cal_active = False
                        self._cal_done = True
                        self._finish_calibration()
                    else:
                        self._flush_smooth()
                        self._cal_step = next_step
                        self._cal_start = time.time()
                        print(f"[Quad Cal] Step {self._cal_step}: Look {self.CAL_STEPS[self._cal_step]}")
                return "CALIBRATING"

            # Keep the current region stable while eyes are closed.
            if eye_open_ratio < self._BLINK_THRESHOLD:
                return self._stable_result(blink=True)

            centered_ud = raw_ud - self._center_baseline_ud
            sv_ud = self._smooth_ud(centered_ud)
            sv_lr = self._smooth_lr(raw_lr)

            self._frame_count += 1
            if self._frame_count < self._WARMUP_FRAMES:
                return "WARMING_UP"

            v_dir = self._classify_vertical(sv_ud)
            h_dir = self._classify_horizontal(sv_lr)
            region = self._combine(v_dir, h_dir)

            self._last_region = region
            self._last_plot_x = float(sv_lr)
            self._last_plot_y = self._normalized_y(sv_ud)

            return {
                "region": region,
                "x": self._last_plot_x,
                "y": self._last_plot_y,
                "blink": blink,
            }

        except Exception as e:
            print("Tracking error:", e)
            return "ERROR"

    # Debug / utility

    def get_debug_info(self):
        ud = float(np.median(self.buf_ud)) if self.buf_ud else 0.0
        lr = float(np.median(self.buf_lr)) if self.buf_lr else 0.5
        return ud, lr, self.UP_T, self.DOWN_T, self.LEFT_T, self.RIGHT_T

    def reset_smoothing(self):
        self._flush_smooth()
        self._reset_hysteresis()