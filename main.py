from fastapi import FastAPI, UploadFile, File, Form
import requests
import os
import base64

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"
FACE_MODEL = "dima806/facial_emotions_image_detection"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Sirf fake positive emotions flip karo
SARCASM_FLIP = {
    "joy": "disgust",
    "surprise": "disgust",
}

# Genuine positive phrases — sarcasm mat lagao inpe
GENUINE_POSITIVE = [
    "got the job", "i love you", "best day", "so happy", "congratulations",
    "got promoted", "we won", "i passed", "she said yes", "he said yes",
    "i got in", "accepted", "i am so excited", "can't wait", "finally did it",
    "i can't believe i got", "best news"
]

# Negative context words — sarcasm zyada likely hai
NEGATIVE_CONTEXT = [
    "ignored", "hate", "stuck", "terrible", "awful", "worst", "again",
    "another monday", "as usual", "obviously", "totally fine", "just great",
    "love being", "love getting", "love waiting", "love sitting",
    "stealing", "talking over", "ghost", "invisible"
]

@app.get("/")
def root():
    return {"message": "EmoLive Backend Running"}

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    text_lower = text.lower()

    # Genuine positive check
    has_genuine_positive = any(phrase in text_lower for phrase in GENUINE_POSITIVE)

    # Negative context check
    has_negative_context = any(phrase in text_lower for phrase in NEGATIVE_CONTEXT)

    # Step 1 — Sarcasm check
    is_sarcastic = False

    if not has_genuine_positive:
        sarcasm_url = f"https://router.huggingface.co/hf-inference/models/{SARCASM_MODEL}"
        sarcasm_response = requests.post(
            sarcasm_url,
            headers=HEADERS,
            json={"inputs": text}
        )
        sarcasm_raw = sarcasm_response.json()
        print(f"Sarcasm raw response: {sarcasm_raw}")

        try:
            if isinstance(sarcasm_raw, list):
                sarcasm_data = sarcasm_raw[0] if isinstance(sarcasm_raw[0], list) else sarcasm_raw
                for item in sarcasm_data:
                    label = item["label"].lower()
                    score = item["score"]

                    threshold = 0.75 if has_negative_context else 0.90

                    if label in ["irony", "sarcasm"] and score > threshold:
                        is_sarcastic = True
                        break
        except:
            is_sarcastic = False
    else:
        print("Genuine positive detected — skipping sarcasm check")

    print(f"Is sarcastic: {is_sarcastic}")

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
    print(f"Emotion raw response: {raw}")

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

        # Step 3 — Sirf joy/surprise flip karo sarcasm mein
        if is_sarcastic:
            top_label = all_scores[0]["label"]
            flipped_label = SARCASM_FLIP.get(top_label, None)

            if flipped_label and flipped_label != top_label:
                original_top_score = all_scores[0]["score"]
                scores_dict = {e["label"]: e["score"] for e in all_scores}
                scores_dict[flipped_label] = original_top_score
                scores_dict[top_label] = 0.01
                all_scores = [{"label": k, "score": v} for k, v in scores_dict.items()]
                all_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "text": text,
            "emotions": [all_scores],
            "is_sarcastic": is_sarcastic
        }

    print(f"Fallback triggered — raw: {raw}")
    return {
        "text": text,
        "emotions": [[{"label": e, "score": 0.0} for e in ALL_EMOTIONS]],
        "is_sarcastic": False
    }

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
