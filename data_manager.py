"""
Session data management and analytics generation.
==================================================
Changes from original data_manager.py
--------------------------------------
save_page_emotion(session_id, page_index, emotion_summary)
    Stores per-region emotion summary alongside gaze analytics when the
    user navigates to the next image set or stops the session.

get_current_page_index(session_id)
    Convenience helper used by app.py to know the current page number
    so the final-page emotion can be tagged correctly.

get_session_summary()
    Now includes per-image emotion data in the returned dict.

All original methods are unchanged.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import gaussian_filter

import config


class DataManager:

    def __init__(self):
        self.sessions: dict = {}
        os.makedirs(config.RESULTS_FOLDER, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Session lifecycle
    # ──────────────────────────────────────────────────────────────────

    def start_session(self, session_id: str, num_images: int,
                      image_names: list[str]):
        self.sessions[session_id] = {
            "id":              session_id,
            "start_time":      datetime.now(),
            "end_time":        None,
            "num_images":      num_images,
            "image_names":     list(image_names),
            "total_blinks":    0,
            "image_data":      self._make_image_data(num_images, image_names),
            "pages":           [],
            "gaze_points_all": [],
        }

    def save_page_analytics(self, session_id: str, page_index: int,
                            analytics: dict):
        """Snapshot gaze-region analytics before moving to the next image set."""
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]

        for region_id, stats in analytics.items():
            if region_id in session["image_data"]:
                d = session["image_data"][region_id]
                d["time_spent"]     = stats.get("time_spent",     0.0)
                d["fixation_count"] = stats.get("fixation_count", 0)

        # Only append page snapshot if it doesn't already exist for this index
        existing_pages = [p["page"] for p in session["pages"]]
        if page_index not in existing_pages:
            session["pages"].append({
                "page":        page_index,
                "image_names": list(session["image_names"]),
                "image_data":  {k: dict(v) for k, v in session["image_data"].items()},
                "emotions":    {},   # will be filled by save_page_emotion
            })
        else:
            # Update existing snapshot with latest gaze data
            for p in session["pages"]:
                if p["page"] == page_index:
                    p["image_data"] = {
                        k: dict(v) for k, v in session["image_data"].items()
                    }
                    break

    def save_page_emotion(self, session_id: str, page_index: int,
                          emotion_summary: dict):
        """
        Store per-region emotion summary for a given page.

        emotion_summary format (from EmotionAnalyzer.get_region_emotion_summary):
        {
          region_id (int): {
            "dominant_emotion": str,
            "avg_scores":       {label: float},
            "sample_count":     int,
          },
          ...
        }

        The summary is attached to the matching page entry so the final
        session report can show e.g. "Image 1 → region 0 → happy".
        """
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]

        # Also merge dominant emotion into image_data for the summary view
        for region_id, estats in emotion_summary.items():
            if region_id in session["image_data"]:
                session["image_data"][region_id]["dominant_emotion"] = \
                    estats.get("dominant_emotion", "unknown")
                session["image_data"][region_id]["emotion_scores"] = \
                    estats.get("avg_scores", {})

        # Find or create the page snapshot and attach emotions
        matching = [p for p in session["pages"] if p["page"] == page_index]
        if matching:
            matching[0]["emotions"] = {
                str(rid): estats for rid, estats in emotion_summary.items()
            }
        else:
            # Page not yet snapshotted (e.g. single-page session stopped early)
            session["pages"].append({
                "page":        page_index,
                "image_names": list(session["image_names"]),
                "image_data":  {k: dict(v) for k, v in session["image_data"].items()},
                "emotions":    {
                    str(rid): estats for rid, estats in emotion_summary.items()
                },
            })

    def get_current_page_index(self, session_id: str) -> int:
        """Return the index of the current (last) page for a session."""
        if session_id not in self.sessions:
            return 0
        return len(self.sessions[session_id].get("pages", []))

    def switch_image_set(self, session_id: str, image_names: list[str]):
        """Register a new image batch; resets per-page counters."""
        if session_id not in self.sessions:
            return
        session = self.sessions[session_id]
        num_images = session["num_images"]
        session["image_names"] = list(image_names)
        session["image_data"]  = self._make_image_data(num_images, image_names)

    def end_session(self, session_id: str, analytics: dict) -> str:
        """Finalise the session and persist the last page's gaze analytics."""
        if session_id not in self.sessions:
            return session_id

        session = self.sessions[session_id]
        session["end_time"] = datetime.now()

        for region_id, stats in analytics.items():
            if region_id in session["image_data"]:
                d = session["image_data"][region_id]
                d["time_spent"]     = stats.get("time_spent",     0.0)
                d["fixation_count"] = stats.get("fixation_count", 0)

        duration = (session["end_time"] - session["start_time"]).total_seconds()
        session["total_duration"] = duration
        return session_id

    # ──────────────────────────────────────────────────────────────────
    # Live update
    # ──────────────────────────────────────────────────────────────────

    def update_session(self, gaze_data: dict):
        session_id = self._current_session_id()
        if session_id is None:
            return
        session = self.sessions[session_id]

        session["total_blinks"] = gaze_data.get("blink_count", 0)

        region = gaze_data.get("region")
        if region is not None and region in session["image_data"]:
            session["image_data"][region].setdefault("hits", 0)
            session["image_data"][region]["hits"] += 1

    # ──────────────────────────────────────────────────────────────────
    # Visualisations
    # ──────────────────────────────────────────────────────────────────

    def generate_heatmap(self, session_id: str,
                         gaze_xy: list[tuple] | None = None) -> str | None:
        if not gaze_xy:
            return None

        width, height = config.HEATMAP_RESOLUTION
        heatmap = np.zeros((height, width))

        for x, y in gaze_xy:
            px = int(x * width)
            py = int(y * height)
            if 0 <= px < width and 0 <= py < height:
                heatmap[py, px] += 1

        heatmap = gaussian_filter(heatmap, sigma=config.HEATMAP_BLUR)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        fig, ax = plt.subplots(figsize=(10, 7.5))
        im = ax.imshow(heatmap, cmap="hot", interpolation="bilinear")
        fig.colorbar(im, ax=ax, label="Attention Intensity")
        ax.set_title(f"Gaze Heatmap – {session_id}")
        ax.axis("off")

        filepath = os.path.join(config.RESULTS_FOLDER,
                                f"{session_id}_overall_heatmap.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def generate_scatter_plot(self, session_id: str,
                              gaze_xy: list[tuple] | None = None,
                              num_images: int | None = None) -> str | None:
        if not gaze_xy:
            return None

        xs, ys = zip(*gaze_xy)

        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.scatter(xs, ys, alpha=0.3, s=10, c="royalblue")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_title(f"Gaze Scatter – {session_id}")
        ax.set_xlabel("X (normalised)")
        ax.set_ylabel("Y (normalised)")
        ax.grid(True, alpha=0.3)

        n   = num_images or (self.sessions.get(session_id, {}).get("num_images", 1))
        lay = config.LAYOUTS.get(n, config.LAYOUTS[1])
        for i in range(1, lay["cols"]):
            ax.axvline(i / lay["cols"], color="red",  linestyle="--", alpha=0.5)
        for i in range(1, lay["rows"]):
            ax.axhline(i / lay["rows"], color="red",  linestyle="--", alpha=0.5)

        filepath = os.path.join(config.RESULTS_FOLDER,
                                f"{session_id}_overall_scatter.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return filepath

    # ──────────────────────────────────────────────────────────────────
    # Summary  (now includes per-image emotion)
    # ──────────────────────────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> dict | None:
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Aggregate per-image gaze + emotion across all pages
        totals: dict[str, dict] = {}
        for page in session.get("pages", []):
            page_emotions = page.get("emotions", {})

            for region_id, d in page["image_data"].items():
                name = d.get("name", f"Image {region_id}")
                if name not in totals:
                    totals[name] = {
                        "name":             name,
                        "time_spent":       0.0,
                        "fixation_count":   0,
                        # Emotion fields – filled below
                        "dominant_emotion": "unknown",
                        "emotion_scores":   {},
                    }
                totals[name]["time_spent"]     += d.get("time_spent",     0.0)
                totals[name]["fixation_count"] += d.get("fixation_count", 0)

                # Attach emotion if available for this region on this page
                estats = page_emotions.get(str(region_id), {})
                if estats:
                    totals[name]["dominant_emotion"] = \
                        estats.get("dominant_emotion", "unknown")
                    totals[name]["emotion_scores"] = \
                        estats.get("avg_scores", {})

        return {
            "session_id":   session_id,
            "duration":     session.get("total_duration", 0.0),
            "total_blinks": session.get("total_blinks", 0),
            "num_images":   session.get("num_images", 1),
            "pages":        len(session.get("pages", [])),
            "images":       list(totals.values()),
        }

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_image_data(num_images: int, image_names: list[str]) -> dict:
        return {
            i: {
                "name":             image_names[i] if i < len(image_names) else f"Image {i}",
                "time_spent":       0.0,
                "fixation_count":   0,
                "dominant_emotion": "unknown",    # ← NEW
                "emotion_scores":   {},           # ← NEW
            }
            for i in range(num_images)
        }

    def _current_session_id(self) -> str | None:
        for sid, s in reversed(list(self.sessions.items())):
            if s.get("end_time") is None:
                return sid
        return None
