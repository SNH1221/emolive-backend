from fastapi import FastAPI, Form
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
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

# Sirf fake positive emotions flip karo
SARCASM_FLIP = {
    "joy": "disapproval",
    "amusement": "annoyance",
    "excitement": "annoyance",
    "optimism": "disappointment",
    "surprise": "annoyance",
}

@app.get("/")
def root():
    return {"message": "EmoLive Backend Running"}

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    text_lower = text.lower()

    has_genuine_positive = any(phrase in text_lower for phrase in GENUINE_POSITIVE)
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
                    threshold = 0.75 if has_negative_context else 0.85
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
            "parameters": {"top_k": 28}
        }
    )

    raw = emotion_response.json()
    print(f"Emotion raw response: {raw}")

    if isinstance(raw, list) and len(raw) > 0:
        emotions_raw = raw[0] if isinstance(raw[0], list) else raw

        all_scores = [
            {"label": e["label"].lower(), "score": e["score"]}
            for e in emotions_raw
        ]

        all_scores.sort(key=lambda x: x["score"], reverse=True)

        # Step 3 — Sarcasm flip
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
        "emotions": [[{"label": "neutral", "score": 1.0}]],
        "is_sarcastic": False
    }
