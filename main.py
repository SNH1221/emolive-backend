from fastapi import FastAPI, UploadFile, File, Form
import requests
import os
import base64

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "michellejieli/emotion_text_classifier"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"
FACE_MODEL = "dima806/facial_emotions_image_detection"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Sarcasm hone pe emotion flip karo
SARCASM_FLIP = {
    "joy": "disgust",
    "disgust": "joy",
    "anger": "joy",
    "sadness": "joy",
    "fear": "neutral",
    "surprise": "neutral",
    "neutral": "neutral"
}

@app.get("/")
def root():
    return {"message": "EmoLive Backend Running"}

# ---------------- TEXT EMOTION ----------------

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    # Step 1 — Sarcasm check
    sarcasm_url = f"https://router.huggingface.co/hf-inference/models/{SARCASM_MODEL}"
    sarcasm_response = requests.post(
        sarcasm_url,
        headers=HEADERS,
        json={"inputs": text}
    )
    sarcasm_raw = sarcasm_response.json()
    
    is_sarcastic = False
    try:
        if isinstance(sarcasm_raw, list):
            sarcasm_data = sarcasm_raw[0] if isinstance(sarcasm_raw[0], list) else sarcasm_raw
            for item in sarcasm_data:
                if item["label"].lower() in ["irony", "sarcasm"] and item["score"] > 0.6:
                    is_sarcastic = True
                    break
    except:
        is_sarcastic = False

    # Step 2 — Emotion detect
    emotion_url = f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"
    emotion_response = requests.post(
        emotion_url,
        headers=HEADERS,
        json={
            "inputs": text,
            "parameters": {"top_k": 7}
        }
    )

    raw = emotion_response.json()

    if isinstance(raw, list) and len(raw) > 0:
        emotions_raw = raw[0] if isinstance(raw[0], list) else raw
        returned_labels = {e["label"].lower(): e["score"] for e in emotions_raw}

        all_scores = []
        for emotion in ALL_EMOTIONS:
            all_scores.append({
                "label": emotion,
                "score": returned_labels.get(emotion, 0.0)
            })

        all_scores.sort(key=lambda x: x["score"], reverse=True)

        # Step 3 — Sarcasm hone pe top emotion flip karo
        if is_sarcastic:
            top_label = all_scores[0]["label"]
            flipped_label = SARCASM_FLIP.get(top_label, top_label)
            # Flipped emotion ko top pe lao
            for e in all_scores:
                if e["label"] == flipped_label:
                    e["score"] = all_scores[0]["score"]
                    all_scores[0]["score"] = 0.01
                    break
            all_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "text": text,
            "emotions": [all_scores],
            "is_sarcastic": is_sarcastic
        }

    return {
        "text": text,
        "emotions": raw,
        "is_sarcastic": False
    }

# ---------------- FACE EMOTION ----------------

@app.post("/detect-face-emotion")
async def detect_face_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    url = f"https://router.huggingface.co/hf-inference/models/{FACE_MODEL}"

    response = requests.post(
        url,
        headers=HEADERS,
        json={"inputs": base64_image}
    )

    return response.json()
