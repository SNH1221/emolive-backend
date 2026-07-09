from fastapi import FastAPI, Form
import requests
import os
import json

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

GENUINE_POSITIVE = [
    "got the job", "i love you", "best day", "so happy", "congratulations",
    "got promoted", "we won", "i passed", "she said yes", "he said yes",
    "i got in", "accepted", "i am so excited", "can't wait", "finally did it",
    "i can't believe i got", "best news", "everything is normal", "doctor said",
    "test results normal", "all clear", "healthy", "recovered", "feeling better"
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

    # Step 3 — Groq final decision
    try:
        top_5 = raw_scores[:5]
        top_5_str = ", ".join([f"{e['label']}({int(e['score']*100)}%)" for e in top_5])
        sarcasm_note = "The text appears to be sarcastic/ironic." if is_sarcastic else ""

        prompt = f"""You are an emotion detection expert. Analyze this text and return corrected emotion scores.

Text: "{text}"
NLP Model detected (top 5): {top_5_str}
{sarcasm_note}

Instructions:
- Consider the full context and meaning
- If sarcastic, adjust emotions accordingly
- Return ONLY a JSON array of top 7 emotions with corrected scores
- Use only these labels: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise
- Scores must sum to 1.0
- For Monday/work frustration, prefer "annoyance" over "disapproval"
- For passive aggressive statements, prefer "annoyance" 
- Return ONLY valid JSON array, no markdown, no explanation

Example: [{{"label": "fear", "score": 0.85}}, {{"label": "nervousness", "score": 0.10}}, {{"label": "surprise", "score": 0.05}}]"""

        groq_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an emotion detection expert. Always respond with valid JSON array only, no explanation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
        )

        groq_data = groq_response.json()
        groq_text = groq_data["choices"][0]["message"]["content"]
        groq_text = groq_text.strip().replace("```json", "").replace("```", "").strip()
        print(f"Groq parsed: {groq_text}")

        final_emotions = json.loads(groq_text)
        final_emotions.sort(key=lambda x: x["score"], reverse=True)

        return {
            "text": text,
            "emotions": [final_emotions],
            "is_sarcastic": is_sarcastic
        }

    except Exception as e:
        print(f"Groq error: {e}")
        return {
            "text": text,
            "emotions": [raw_scores],
            "is_sarcastic": is_sarcastic
        }

@app.post("/generate-ai-response")
async def generate_ai_response(emotion: str = Form(...)):
    try:
        prompt = when_emotion_prompt(emotion)

        groq_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a compassionate emotional support assistant. Give warm, empathetic responses in 2-3 sentences."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
        )

        data = groq_response.json()
        ai_text = data["choices"][0]["message"]["content"]
        return {"response": ai_text}

    except Exception as e:
        print(f"AI Response error: {e}")
        return {"response": "Could not generate response."}

def when_emotion_prompt(emotion: str) -> str:
    emotion = emotion.lower()
    if emotion in ["joy", "excitement", "admiration"]:
        return f"The user is feeling {emotion}. Give a warm, celebratory empathetic response in 2-3 sentences. Suggest one way to maintain this positive energy."
    elif emotion in ["sadness", "grief", "disappointment", "remorse"]:
        return f"The user is feeling {emotion}. Give a compassionate empathetic response in 2-3 sentences. Suggest one gentle positive reframe."
    elif emotion in ["anger", "annoyance", "disapproval"]:
        return f"The user is feeling {emotion}. Give a calm, understanding empathetic response in 2-3 sentences. Suggest one way to channel this energy positively."
    elif emotion in ["fear", "nervousness"]:
        return f"The user is feeling {emotion}. Give a reassuring empathetic response in 2-3 sentences. Suggest one grounding technique."
    elif emotion == "relief":
        return f"The user is feeling relief. Give a warm, validating empathetic response in 2-3 sentences."
    elif emotion in ["love", "caring"]:
        return f"The user is feeling {emotion}. Give a warm, heartfelt empathetic response in 2-3 sentences."
    elif emotion == "confusion":
        return f"The user is feeling confused. Give a calm, clarifying empathetic response in 2-3 sentences."
    elif emotion == "surprise":
        return f"The user is feeling surprised. Give a fun, engaging empathetic response in 2-3 sentences."
    else:
        return f"The user is feeling {emotion}. Give a gentle, encouraging empathetic response in 2-3 sentences."
