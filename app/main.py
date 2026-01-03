from fastapi import FastAPI, HTTPException
from typing import List, Dict
import numpy as np

from session_vector import build_session_vector
from fingerprint import build_fingerprint
from anomaly_score import anomaly_score

app = FastAPI(title="SentinelX MVP", version="1.0")

FINGERPRINT = None
THRESHOLD = 3.0  # conservative anomaly threshold

@app.post("/train")
def train(sessions: List[List[Dict]]):
    """
    Train behavioral fingerprint from normal sessions.
    """
    global FINGERPRINT

    if len(sessions) < 5:
        raise HTTPException(status_code=400, detail="Need more training sessions")

    vectors = [build_session_vector(s) for s in sessions]
    FINGERPRINT = build_fingerprint(vectors)

    return {"status": "trained", "sessions_used": len(sessions)}

@app.post("/analyze")
def analyze(session: List[Dict]):
    """
    Analyze a session and return anomaly score.
    """
    if FINGERPRINT is None:
        raise HTTPException(status_code=400, detail="Model not trained")

    vector = build_session_vector(session)
    score = anomaly_score(vector, FINGERPRINT)

    return {
        "anomaly_score": float(score),
        "verdict": "suspicious" if score > THRESHOLD else "normal"
    }
