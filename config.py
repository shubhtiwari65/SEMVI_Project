"""
Configuration settings for Eye Tracking Analysis System
"""

# ============ CAMERA & MEDIAPIPE ============
CAMERA_INDEX = 0
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE  = 0.6

# ============ EYE LANDMARKS ============
LEFT_IRIS  = 468
RIGHT_IRIS = 473
LEFT_EYE   = [159, 145, 133, 33]    # top, bottom, left, right
RIGHT_EYE  = [386, 374, 362, 263]

# ============ BLINK DETECTION ============
BLINK_THRESHOLD       = 0.26  # Eye Aspect Ratio below this = closed  (tune 0.22-0.30)
BLINK_DEBOUNCE_FRAMES = 3     # Frames to skip after a blink is registered

# ============ GAZE TRACKING ============
CALIBRATION_FRAMES = 60   # Frames of neutral gaze used for calibration average
GAIN_X        = 2.8       # Horizontal gaze sensitivity multiplier
GAIN_Y        = 2.3     # Vertical gaze sensitivity multiplier
SMOOTH_FACTOR = 0.22      # EMA weight – lower = smoother but slower (0.05-0.20)
DEAD_ZONE     = 0.03     # Delta below which gaze movement is ignored

# ============ REGION DETECTION ============
REGION_HISTORY_SIZE  = 25  # Rolling window size (frames) for majority-vote
CONFIDENCE_THRESHOLD = 0.65 # Fraction of window that must agree on one region
STABLE_FRAMES        = 8  # Consecutive frames the winning region must hold
                             # before a region-change event is emitted

# ============ GRID LAYOUTS ============
LAYOUTS = {
    1: {"rows": 1, "cols": 1, "regions": 1},
    2: {"rows": 1, "cols": 2, "regions": 2},
    4: {"rows": 2, "cols": 2, "regions": 4},
}

# ============ ANALYTICS ============
MIN_DWELL_TIME     = 0.15        # Minimum seconds to count a region visit
HEATMAP_RESOLUTION = (800, 600)  # Width × height of generated heatmap images
HEATMAP_BLUR       = 50          # Gaussian sigma applied to heatmap density

# ============ FLASK ============
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False
CORS_ALLOWED_ORIGINS = "*"

# ============ FILE PATHS ============
IMAGE_FOLDER   = "static/images"
RESULTS_FOLDER = "static/results"
