import cv2
from deepface import DeepFace
import threading

emotion_result = "Waiting..."
running = False

def webcam_loop():
    global emotion_result, running

    cap = cv2.VideoCapture(0)
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            emotion_result = result[0]["dominant_emotion"]
        except:
            emotion_result = "No face"

    cap.release()

def start_detection():
    global running
    if not running:
        thread = threading.Thread(target=webcam_loop, daemon=True)
        thread.start()

def stop_detection():
    global running
    running = False

def get_emotion():
    return emotion_result
