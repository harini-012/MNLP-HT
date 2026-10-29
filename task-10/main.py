
import re
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "spam_model.joblib"
vectorizer_path = BASE_DIR / "tfidf_vectorizer.joblib"

if not model_path.exists() or not vectorizer_path.exists():
    raise RuntimeError(
        "Model files not found. Run: python train_model.py"
    )

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI(
    title="SMS Spam Detector API",
    description="Classifies SMS messages as spam or ham.",
    version="1.0.0",
)


class SMSRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="SMS message to classify",
    )


class SMSResponse(BaseModel):
    prediction: str
    confidence: float


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.get("/")
def home():
    return {
        "message": "SMS Spam Detector API is running",
        "docs": "/docs",
    }


@app.post("/predict", response_model=SMSResponse)
def predict_sms(request: SMSRequest):
    message = clean_text(request.message)

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message must contain readable text.",
        )

    features = vectorizer.transform([message])

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])

    label = "spam" if prediction == 1 else "ham"

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
    }
