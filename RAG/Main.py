import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - depends on local env
    def load_dotenv():
        return False

try:
    from google import genai
except ImportError:  # pragma: no cover - depends on local env
    genai = None


RAG_QUESTIONS = [
    "Which images held the user's attention the longest, and what does that suggest about focus?",
    "How consistent was the user's gaze behavior across the session?",
    "How well did the user's facial emotion match the target emotion overall?",
    "Were there any repeated mismatch patterns between target emotion and detected emotion?",
    "What does the blink count suggest about comfort, effort, or fatigue during the session?",
    "What is the most important takeaway from this session for the user?",
    "What is one practical suggestion that could improve the next session?",
]
DEFAULT_QUESTION = "Create a natural, human-friendly session report using the analysis questions."
_CLIENT = None


def _project_root():
    return Path(__file__).resolve().parents[1]


def _load_summary(json_path):
    report_path = Path(json_path)
    if not report_path.is_absolute():
        report_path = (_project_root() / report_path).resolve()

    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file), report_path


def _session_stats(summary):
    metadata = summary.get("metadata", {})
    interactions = summary.get("interactions", [])
    total = len(interactions)
    matches = sum(1 for item in interactions if item.get("matched"))
    mismatches = total - matches
    total_blinks = metadata.get("total_blinks", 0)
    total_duration = float(metadata.get("total_duration_seconds", 0) or 0)

    gaze_counts = {}
    emotion_mismatches = {}
    duration_sorted = sorted(
        interactions,
        key=lambda item: float(item.get("duration_seconds", 0) or 0),
        reverse=True,
    )

    for item in interactions:
        gaze = item.get("gaze_region") or "unknown"
        gaze_counts[gaze] = gaze_counts.get(gaze, 0) + 1

        target = item.get("target_emotion") or "unknown"
        user = item.get("user_emotion") or "unknown"
        if target != "unknown" and user != "unknown" and target != user:
            key = f"{target} -> {user}"
            emotion_mismatches[key] = emotion_mismatches.get(key, 0) + 1

    top_gaze = max(gaze_counts, key=gaze_counts.get) if gaze_counts else "unknown"
    top_image = duration_sorted[0]["image"] if duration_sorted else "unknown"
    top_image_duration = float(duration_sorted[0].get("duration_seconds", 0) or 0) if duration_sorted else 0
    dominant_mismatch = (
        max(emotion_mismatches, key=emotion_mismatches.get) if emotion_mismatches else None
    )
    match_rate = round((matches / total) * 100, 1) if total else 0.0

    return {
        "metadata": metadata,
        "interactions": interactions,
        "total": total,
        "matches": matches,
        "mismatches": mismatches,
        "match_rate": match_rate,
        "total_blinks": total_blinks,
        "total_duration": total_duration,
        "top_gaze": top_gaze,
        "top_image": top_image,
        "top_image_duration": top_image_duration,
        "dominant_mismatch": dominant_mismatch,
        "gaze_counts": gaze_counts,
    }


def _build_context(summary):
    stats = _session_stats(summary)
    metadata = stats["metadata"]
    interactions = stats["interactions"]

    if not interactions:
        return (
            f"Session ID: {metadata.get('session_id', 'unknown')}\n"
            f"Duration: {metadata.get('total_duration_seconds', 0)} seconds\n"
            f"Total blinks: {metadata.get('total_blinks', 0)}\n"
            "No image interactions were recorded."
        )

    lines = [
        f"Session ID: {metadata.get('session_id', 'unknown')}",
        f"Duration: {stats['total_duration']} seconds",
        f"Images interacted: {stats['total']}",
        f"Emotion matches: {stats['matches']}",
        f"Emotion mismatches: {stats['mismatches']}",
        f"Match rate: {stats['match_rate']}%",
        f"Total blinks: {stats['total_blinks']}",
        f"Most frequent gaze region: {stats['top_gaze']}",
        f"Longest-viewed image: {stats['top_image']} ({stats['top_image_duration']}s)",
    ]

    if stats["dominant_mismatch"]:
        lines.append(f"Most repeated mismatch pattern: {stats['dominant_mismatch']}")

    for idx, item in enumerate(interactions, start=1):
        lines.append(
            (
                f"{idx}. Image={item.get('image', 'unknown')}, "
                f"Target emotion={item.get('target_emotion', 'unknown')}, "
                f"User emotion={item.get('user_emotion', 'unknown')}, "
                f"Matched={item.get('matched', False)}, "
                f"Gaze={item.get('gaze_region', 'unknown')}, "
                f"Duration={item.get('duration_seconds', 0)}s, "
                f"Views={item.get('views', 0)}"
            )
        )

    return "\n".join(lines)


def _build_fallback_answer(summary):
    stats = _session_stats(summary)
    if not stats["interactions"]:
        return "No image interaction data was recorded in this session, so there is not enough evidence for a meaningful RAG summary yet."

    attention_line = (
        f"The strongest attention was on {stats['top_image']}, where the user spent about "
        f"{stats['top_image_duration']:.1f} seconds, and the most common gaze region was "
        f"{stats['top_gaze']}."
    )

    emotion_line = (
        f"Emotion matching was {stats['match_rate']:.1f}%, with {stats['matches']} successful "
        f"match(es) out of {stats['total']} viewed image(s)."
    )
    if stats["dominant_mismatch"]:
        emotion_line += f" The most repeated mismatch pattern was {stats['dominant_mismatch']}."

    if stats["total_blinks"] <= 2:
        blink_line = (
            f"The blink count was low at {stats['total_blinks']}, which may suggest a fairly steady level of visual concentration."
        )
    elif stats["total_blinks"] <= 8:
        blink_line = (
            f"The session recorded {stats['total_blinks']} blinks, which suggests a moderate amount of effort without obvious overload."
        )
    else:
        blink_line = (
            f"The session recorded {stats['total_blinks']} blinks, which may indicate rising effort or fatigue as the task progressed."
        )

    suggestion = (
        "For the next session, it would help to spend a little more time on each image while intentionally mirroring the target facial expression before shifting gaze."
        if stats["mismatches"] > 0
        else "For the next session, keep the same steady pace and try repeating the strongest-matching expressions to reinforce consistency."
    )

    return " ".join([attention_line, emotion_line, blink_line, suggestion])


def _get_client():
    global _CLIENT

    if _CLIENT is not None:
        return _CLIENT

    if genai is None:
        return None

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _build_prompt(summary, question):
    context = _build_context(summary)
    formatted_questions = "\n".join(f"- {item}" for item in RAG_QUESTIONS)
    return f"""
You are analyzing a copy-emotion eye-tracking session and writing feedback for a real user.

Session context:
{context}

Primary task:
{question}

Answer all of these analysis questions internally before writing the final report:
{formatted_questions}

Writing instructions:
- Write a natural, humanized report in 4 short paragraphs.
- Keep the tone supportive, specific, and easy to understand.
- Focus on attention behavior, emotional matching, blink interpretation, and one practical improvement.
- Do not sound robotic, repetitive, or overly clinical.
- Do not mention that you were asked multiple questions.
- If the session is weak or incomplete, explain that gently and constructively.
"""


def generate_rag_answer_from_summary(summary, question=DEFAULT_QUESTION):
    client = _get_client()
    if client is None:
        return {
            "answer": _build_fallback_answer(summary),
            "source": "fallback",
            "error": "Gemini client is unavailable. Check google-genai and GEMINI_API_KEY.",
            "questions_used": RAG_QUESTIONS,
        }

    prompt = _build_prompt(summary, question)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = (response.text or "").strip()
        if not answer:
            raise ValueError("Empty response from Gemini")
        return {
            "answer": answer,
            "source": "gemini",
            "error": None,
            "questions_used": RAG_QUESTIONS,
        }
    except Exception as exc:  # pragma: no cover - external API path
        return {
            "answer": _build_fallback_answer(summary),
            "source": "fallback",
            "error": str(exc),
            "questions_used": RAG_QUESTIONS,
        }


def generate_rag_answer_from_json(json_path, question=DEFAULT_QUESTION):
    summary, report_path = _load_summary(json_path)
    result = generate_rag_answer_from_summary(summary, question=question)
    result["json_path"] = str(report_path)
    return result


if __name__ == "__main__":
    import sys

    target_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else _project_root() / "static" / "results" / "session_1776448583_RAG_report.json"
    )

    result = generate_rag_answer_from_json(target_path)
    print(result["answer"])
    if result.get("error"):
        print(f"[RAG warning] {result['error']}")
