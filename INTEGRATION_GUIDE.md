# Emotion + Gaze Integration Guide
=====================================

## What Changed and Why

### New file: `emotion_analyzer.py`
The original `backend_emotion.py` opened its own `VideoCapture(0)`, which
would CONFLICT with `app.py` already holding the camera. This new module
instead receives the frame directly from the tracking loop — no second camera
is opened. DeepFace runs in its own daemon thread so it never blocks the
~100 Hz gaze loop.

### Modified: `app.py`
- Imports `EmotionAnalyzer`
- `tracking_loop` calls `emotion_analyzer.analyze(frame, current_region)` 
  after every gaze frame
- `gaze_update` socket event now carries 3 new fields:
    - `emotion`        → smoothed dominant emotion for the current region
    - `emotion_scores` → raw DeepFace probability dict {label: float}
    - `emotion_region` → which region the emotion was captured for
- `/api/navigate` snapshots and resets emotion per page (same as gaze)
- `/api/stop_session` returns `emotion_summary` per region
- `/api/recalibrate` also clears emotion history

### Modified: `data_manager.py`
- `_make_image_data` seeds `dominant_emotion` and `emotion_scores` per region
- New method `save_page_emotion(session_id, page_index, emotion_summary)`
- New helper `get_current_page_index(session_id)`
- `get_session_summary` now includes `dominant_emotion` and `emotion_scores`
  per image in the returned `"images"` list

---

## Frontend: Handling Emotion in the SocketIO `gaze_update` Event

```javascript
// In your existing socket.on("gaze_update", ...) handler, add:

socket.on("gaze_update", (data) => {
  // ── Existing gaze fields (unchanged) ─────────────────────────────
  const region        = data.region;           // int | null
  const confidence    = data.confidence;       // 0.0 – 1.0
  const blinkCount    = data.blink_count;      // int
  const eyesClosed    = data.eyes_closed;      // bool
  const regionChanged = data.region_changed;   // bool

  // ── NEW emotion fields ────────────────────────────────────────────
  const emotion       = data.emotion;          // e.g. "happy"
  const emotionScores = data.emotion_scores;   // {happy:92.1, sad:1.2, ...}
  const emotionRegion = data.emotion_region;   // which region it belongs to

  // Example: update a UI badge per region
  if (emotionRegion !== null && emotionRegion !== undefined) {
    const badge = document.getElementById(`emotion-badge-${emotionRegion}`);
    if (badge) {
      badge.textContent = emotion;
      badge.className   = `emotion-badge emotion-${emotion}`;
    }
  }

  // Example: show a global emotion indicator
  document.getElementById("current-emotion").textContent =
    `Region ${emotionRegion}: ${emotion}`;
});
```

### Suggested HTML elements

```html
<!-- Add one badge per image region slot -->
<div id="region-0" class="image-slot">
  <img src="..." />
  <span id="emotion-badge-0" class="emotion-badge">Detecting...</span>
</div>

<div id="region-1" class="image-slot">
  <img src="..." />
  <span id="emotion-badge-1" class="emotion-badge">Detecting...</span>
</div>

<!-- Global status bar -->
<div id="current-emotion" class="emotion-status"></div>
```

### Suggested CSS

```css
.emotion-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 0.85rem;
  background: rgba(0,0,0,0.55);
  color: #fff;
  transition: background 0.4s ease;
}
.emotion-happy    { background: #f59e0b; }
.emotion-sad      { background: #3b82f6; }
.emotion-angry    { background: #ef4444; }
.emotion-surprise { background: #a855f7; }
.emotion-fear     { background: #6b7280; }
.emotion-disgust  { background: #10b981; }
.emotion-neutral  { background: #94a3b8; }
```

---

## Session Summary: Emotion per Image

After `POST /api/stop_session` the response includes:

```json
{
  "summary": {
    "images": [
      {
        "name": "photo_a.jpg",
        "time_spent": 12.3,
        "fixation_count": 8,
        "dominant_emotion": "happy",
        "emotion_scores": { "happy": 74.2, "neutral": 18.1, ... }
      },
      ...
    ]
  },
  "emotion_summary": {
    "0": { "dominant_emotion": "happy",   "avg_scores": {...}, "sample_count": 42 },
    "1": { "dominant_emotion": "neutral", "avg_scores": {...}, "sample_count": 37 }
  }
}
```

---

## Files to deploy

| File                   | Status             |
|------------------------|--------------------|
| `emotion_analyzer.py`  | NEW — add to project |
| `app.py`               | REPLACE existing   |
| `data_manager.py`      | REPLACE existing   |
| `gaze_analyzer.py`     | UNCHANGED          |
| `config.py`            | UNCHANGED          |
| `backend_emotion.py`   | RETIRED — no longer needed |

## Dependencies

Make sure DeepFace is installed:
```
pip install deepface
```
Everything else (OpenCV, MediaPipe, Flask-SocketIO) was already present.
