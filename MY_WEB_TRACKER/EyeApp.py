import os
import time
import threading
import math
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- MEDIAPIPE SETUP ---
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants
LEFT_IRIS = 468
LEFT_KEY = [159, 145, 133, 33]
RIGHT_IRIS = 473
RIGHT_KEY = [386, 374, 362, 263]

# --- CALIBRATION ZONE (ADJUST THESE NUMBERS!) ---
# READ THE TERMINAL TO FIND YOUR PERFECT VALUES
# If the dot doesn't reach the left side, INCREASE HOR_MIN
# If the dot doesn't reach the right side, DECREASE HOR_MAX
HOR_MIN = 0.40   # Typical range: 0.35 to 0.45
HOR_MAX = 0.60   # Typical range: 0.55 to 0.65

VER_MIN = 0.30   # Typical range: 0.25 to 0.35
VER_MAX = 0.45   # Typical range: 0.40 to 0.50

# SMOOTHING (Higher = Slower/Smoother, Lower = Faster/Jittery)
# We use a moving average of the last 'N' frames
SMOOTH_HISTORY = 5 
x_history = deque(maxlen=SMOOTH_HISTORY)
y_history = deque(maxlen=SMOOTH_HISTORY)

BLINK_THRESHOLD = 0.45

# Global State
blink_counter = 0
blink_counter_frame = 0

def process_eye(landmarks, iris_id, key_points):
    def get_pt(idx):
        return landmarks[idx].x, landmarks[idx].y

    p_iris = get_pt(iris_id)
    p_top = get_pt(key_points[0])
    p_bottom = get_pt(key_points[1])
    p_left = get_pt(key_points[2])
    p_right = get_pt(key_points[3])

    # Blink Ratio
    v_len = math.hypot(p_top[0]-p_bottom[0], p_top[1]-p_bottom[1])
    h_len = math.hypot(p_left[0]-p_right[0], p_left[1]-p_right[1])
    blink_ratio = v_len / h_len if h_len != 0 else 1.0

    # Gaze Ratios
    dist_h = p_iris[0] - p_left[0]
    total_w = p_right[0] - p_left[0]
    gaze_h = dist_h / total_w if total_w != 0 else 0.5

    dist_v = p_iris[1] - p_top[1]
    total_h = p_bottom[1] - p_top[1]
    gaze_v = dist_v / total_h if total_h != 0 else 0.5

    return blink_ratio, gaze_h, gaze_v

def tracker_loop():
    global blink_counter, blink_counter_frame
    
    print(" [TRACKER] Starting Camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(" [ERROR] No camera found.")
        return

    print(" [TRACKER] Active. Look at the terminal to CALIBRATE.")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1) 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)
            
            if output.multi_face_landmarks:
                lm = output.multi_face_landmarks[0].landmark
                
                l_blink, l_h, l_v = process_eye(lm, LEFT_IRIS, LEFT_KEY)
                r_blink, r_h, r_v = process_eye(lm, RIGHT_IRIS, RIGHT_KEY)

                # --- BLINK ---
                if l_blink < BLINK_THRESHOLD and r_blink < BLINK_THRESHOLD:
                    if blink_counter_frame == 0:
                        blink_counter += 1
                        blink_counter_frame = 1
                        socketio.emit('blink_update', {'count': blink_counter})
                else:
                    if blink_counter_frame > 0:
                        blink_counter_frame += 1
                        if blink_counter_frame > 5: blink_counter_frame = 0

                # --- GAZE ACCURACY LOGIC ---
                avg_h = (l_h + r_h) / 2
                avg_v = (l_v + r_v) / 2

                # 1. Map to Screen (with constraints)
                # Note: We invert X (1.0, 0.0) because of the mirror effect
                raw_x = np.interp(avg_h, [HOR_MIN, HOR_MAX], [1.0, 0.0])
                raw_y = np.interp(avg_v, [VER_MIN, VER_MAX], [0.0, 1.0])

                # 2. Add to History for Smoothing
                x_history.append(raw_x)
                y_history.append(raw_y)

                # 3. Calculate Average of last N frames
                smooth_x = sum(x_history) / len(x_history)
                smooth_y = sum(y_history) / len(y_history)

                socketio.emit('gaze_update', {'x': smooth_x, 'y': smooth_y})

                # --- CALIBRATION PRINTER (IMPORTANT!) ---
                # Print this so you can tune your HOR_MIN / HOR_MAX
                print(f"RAW GAZE -> H: {avg_h:.3f} | V: {avg_v:.3f} || Current Settings: {HOR_MIN}-{HOR_MAX}", end='\r')

            time.sleep(0.01)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    t = threading.Thread(target=tracker_loop, daemon=True)
    t.start()
    socketio.run(app, debug=True, use_reloader=False)