from flask import Flask, render_template, jsonify
import os
from backend_emotion import start_detection, stop_detection, get_emotion

app = Flask(__name__)

IMAGE_FOLDER = os.path.join("static", "images")

@app.route("/")
def index():
    images = os.listdir(IMAGE_FOLDER)
    images.sort()
    return render_template("index.html", images=images)

@app.route("/start")
def start():
    start_detection()
    return jsonify({"status": "started"})

@app.route("/stop")
def stop():
    stop_detection()
    return jsonify({"status": "stopped"})

@app.route("/emotion")
def emotion():
    return jsonify({"emotion": get_emotion()})

if __name__ == "__main__":
    print("APP.PY STARTED")
    app.run(debug=True)
