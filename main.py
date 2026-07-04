from fastapi import FastAPI, Form
import requests
import os
import json

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

GENUINE_POSITIVE = [
    "got the job", "i love you", "best day", "so happy", "congratulations",
    "got promoted", "we won", "i passed", "she said yes", "he said yes",
    "i got in", "accepted", "i am so excited", "can't wait", "finally did it",
    "i can't believe i got", "best news"
]

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
    has_genuine_positive = any(phrase in text_lower for phrase in GENUINE_POSITIVE)
    has_negative_context = any(phrase in text_lower for phrase in NEGATIVE_CONTEXT)

    # Step 1 — Sarcasm check
    is_sarcastic = False
    if not has_genuine_positive:
        try:
            sarcasm_url = f"https://router.huggingface.co/hf-inference/models/{SARCASM_MODEL}"
            sarcasm_response = requests.post(
                sarcasm_url,
                headers=HEADERS,
                json={"inputs": text}
            )
            sarcasm_raw = sarcasm_response.json()
            print(f"Sarcasm raw: {sarcasm_raw}")

            if isinstance(sarcasm_raw, list):
                sarcasm_data = sarcasm_raw[0] if isinstance(sarcasm_raw[0], list) else sarcasm_raw
                for item in sarcasm_data:
                    threshold = 0.75 if has_negative_context else 0.85
                    if item["label"].lower() in ["irony", "sarcasm"] and item["score"] > threshold:
                        is_sarcastic = True
                        break
        except:
            is_sarcastic = False

    print(f"Is sarcastic: {is_sarcastic}")

    # Step 2 — Emotion detect
    try:
        emotion_url = f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"
        emotion_response = requests.post(
            emotion_url,
            headers=HEADERS,
            json={"inputs": text, "parameters": {"top_k": 28}}
        )
        raw = emotion_response.json()
        print(f"Emotion raw: {raw}")

        if isinstance(raw, list) and len(raw) > 0:
            emotions_raw = raw[0] if isinstance(raw[0], list) else raw
            raw_scores = [
                {"label": e["label"].lower(), "score": round(e["score"], 4)}
                for e in emotions_raw
            ]
            raw_scores.sort(key=lambda x: x["score"], reverse=True)
        else:
            raw_scores = [{"label": "neutral", "score": 1.0}]
    except:
        raw_scores = [{"label": "neutral", "score": 1.0}]

    # Step 3 — Gemini final decision
    try:
        top_5 = raw_scores[:5]
        top_5_str = ", ".join([f"{e['label']}({int(e['score']*100)}%)" for e in top_5])

        sarcasm_note = "The text appears to be sarcastic/ironic." if is_sarcastic else ""

        prompt = f"""You are an emotion detection expert. Analyze this text and return the corrected emotion scores.

Text: "{text}"
NLP Model detected (top 5): {top_5_str}
{sarcasm_note}

Instructions:
- Consider the full context and meaning of the text
- If sarcastic, adjust emotions accordingly
- Return ONLY a JSON array of top 7 emotions with corrected scores
- Use these emotion labels only: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise
- Scores must sum to 1.0
- Return ONLY the JSON array, no other text

Example format:
[{{"label": "joy", "score": 0.85}}, {{"label": "excitement", "score": 0.10}}, {{"label": "relief", "score": 0.05}}]"""

        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

        gemini_response = requests.post(
            gemini_url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1}
            }
        )

        gemini_data = gemini_response.json()
        gemini_text = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
        gemini_text = gemini_text.strip().replace("```json", "").replace("```", "").strip()

        print(f"Gemini response: {gemini_text}")

        final_emotions = json.loads(gemini_text)
        final_emotions.sort(key=lambda x: x["score"], reverse=True)

        return {
            "text": text,
            "emotions": [final_emotions],
            "is_sarcastic": is_sarcastic
        }

    except Exception as e:
        print(f"Gemini error: {e} — falling back to NLP scores")
        return {
            "text": text,
            "emotions": [raw_scores],
            "is_sarcastic": is_sarcastic
        }
