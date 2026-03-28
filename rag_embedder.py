"""
rag_embedder.py
----------------
Takes the structured JSON output from the Eye Tracking & Emotion app, 
translates it into semantic behavioral descriptions, and generates 
vector embeddings for use in a health-analysis RAG model.
"""

import json
from sentence_transformers import SentenceTransformer

class BehavioralEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the embedding model. 
        'all-MiniLM-L6-v2' is fast, lightweight, and standard for local RAG testing.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    def textualize_session_data(self, session_summary: dict) -> list[dict]:
        """
        Converts the raw JSON session summary into semantic text chunks 
        optimized for similarity search and LLM context comprehension.
        """
        chunks = []
        
        session_id = session_summary.get("session_id", "Unknown")
        duration = session_summary.get("duration", 0.0)
        total_blinks = session_summary.get("total_blinks", 0)
        
        # Calculate blink rate (blinks per minute) - useful for neurological/fatigue analysis
        bpm = (total_blinks / duration) * 60 if duration > 0 else 0

        # Chunk 1: Global Session Baseline
        global_text = (
            f"Session ID {session_id} overview: The subject was monitored for {duration:.1f} seconds. "
            f"During this time, the subject blinked a total of {total_blinks} times, "
            f"resulting in an average blink rate of {bpm:.1f} blinks per minute."
        )
        chunks.append({
            "chunk_id": f"{session_id}_global",
            "metadata": {"session_id": session_id, "type": "global_baseline"},
            "text": global_text
        })

        # Chunks 2+: Per-Stimulus (Image) Behavioral Reactions
        images = session_summary.get("images", [])
        for img in images:
            name = img.get("name", "Unknown Image")
            time_spent = img.get("time_spent", 0.0)
            fixations = img.get("fixation_count", 0)
            dom_emotion = img.get("dominant_emotion", "unknown")
            scores = img.get("emotion_scores", {})
            
            # Extract the top 2 emotions for nuance
            sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            emotion_details = ""
            if len(sorted_emotions) > 0:
                emotion_details = f"The primary emotion detected was {dom_emotion} ({sorted_emotions[0][1]:.1f}% confidence)."
            if len(sorted_emotions) > 1:
                emotion_details += f" Secondary emotional indicators included {sorted_emotions[1][0]} ({sorted_emotions[1][1]:.1f}%)."

            img_text = (
                f"Stimulus Reaction for {name} during Session {session_id}: "
                f"The subject spent {time_spent:.1f} seconds observing this visual stimulus, "
                f"registering {fixations} distinct gaze fixations. "
                f"Emotionally, the subject presented as predominantly {dom_emotion}. "
                f"{emotion_details}"
            )
            
            chunks.append({
                "chunk_id": f"{session_id}_stimulus_{name}",
                "metadata": {
                    "session_id": session_id, 
                    "type": "stimulus_reaction",
                    "stimulus_name": name,
                    "dominant_emotion": dom_emotion
                },
                "text": img_text
            })

        return chunks

    def process_and_embed(self, session_summary: dict) -> list[dict]:
        """
        Takes the JSON summary, creates text chunks, and attaches vector embeddings.
        This output can be directly pushed to Pinecone, Weaviate, Milvus, or FAISS.
        """
        chunks = self.textualize_session_data(session_summary)
        
        # Extract just the text to pass to the embedding model
        texts_to_embed = [chunk["text"] for chunk in chunks]
        
        # Generate vector embeddings
        embeddings = self.model.encode(texts_to_embed)
        
        # Reattach embeddings to the payload
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist() # Convert numpy array to list for JSON serialization
            
        return chunks

# ==========================================
# Example Usage for the Downstream Developer
# ==========================================
if __name__ == "__main__":
    # Mock data formatted exactly how app.py outputs it
    mock_session_output = {
        "session_id": "session_20260323_231941",
        "duration": 145.2,
        "total_blinks": 42,
        "num_images": 2,
        "images": [
            {
                "name": "calming_landscape.jpg",
                "time_spent": 45.0,
                "fixation_count": 12,
                "dominant_emotion": "neutral",
                "emotion_scores": {"neutral": 85.1, "happy": 10.2, "sad": 2.1}
            },
            {
                "name": "stress_test_medical.jpg",
                "time_spent": 80.5,
                "fixation_count": 35,
                "dominant_emotion": "fear",
                "emotion_scores": {"fear": 65.4, "surprise": 20.1, "sad": 5.0}
            }
        ]
    }

    embedder = BehavioralEmbedder()
    vector_db_payload = embedder.process_and_embed(mock_session_output)

    # Print the result to verify
    for item in vector_db_payload:
        print(f"\n--- Chunk: {item['chunk_id']} ---")
        print(f"Text: {item['text']}")
        print(f"Vector Dimensions: {len(item['embedding'])}") # Will be 384 for MiniLM
        print(f"Vector Preview: {item['embedding'][:3]} ...")