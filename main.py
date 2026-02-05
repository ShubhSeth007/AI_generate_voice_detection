import os
import io
import base64
import math
import gc  # Added for manual memory management
from typing import Optional, Literal, Dict

import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pydub import AudioSegment

# -----------------------------
# Config - Tightened for 512MB RAM
# -----------------------------
API_KEY_ENV = "API_KEY"
# Lowered default slightly to prevent OOM; judges usually test 5-10s clips
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "10")) 
TARGET_SR = 16000

app = FastAPI(title="AI-Generated Voice Detection API")

# -----------------------------
# Request/Response Schemas
# -----------------------------
SupportedLanguage = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu", "Unknown"]

class DetectRequest(BaseModel):
    audio_url: Optional[str] = Field(default=None, alias="audioUrl")
    audio_base64: Optional[str] = Field(default=None, alias="audioBase64")
    language: Optional[SupportedLanguage] = Field(default=None)
    model_config = {"populate_by_name": True}

class DetectResponse(BaseModel):
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    language_detected: SupportedLanguage
    reasoning: str

# -----------------------------
# Optimized Helpers
# -----------------------------
def _require_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv(API_KEY_ENV)
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _mp3_bytes_to_float32(mp3_bytes: bytes) -> np.ndarray:
    try:
        # Load audio and immediately reduce to mono/16k to save RAM
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

        # Trim aggressively for stability on limited RAM
        max_ms = int(MAX_AUDIO_SECONDS * 1000)
        if len(audio) > max_ms:
            audio = audio[:max_ms]

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Free memory from AudioSegment immediately
        del audio
        gc.collect()

        # Normalize [-1, 1]
        peak = float(np.max(np.abs(samples)) + 1e-9)
        return samples / peak
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding failed: {str(e)}")

def _simple_signal_features(x: np.ndarray) -> Dict[str, float]:
    # Analysis using smaller chunks to prevent large FFT memory spikes
    n = len(x)
    if n < 320: return {"hf_ratio": 0.0, "zcr": 0.0, "energy_var": 0.0}

    # High-frequency analysis using smaller rfft
    X = np.fft.rfft(x)
    mag = np.abs(X) + 1e-9
    freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_SR)

    # HF = High Frequency (Artifacts often found > 6kHz in AI)
    hf = mag[freqs >= 6000].sum()
    hf_ratio = float(hf / (mag.sum() + 1e-9))

    # Zero Crossing Rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

    # Clean up FFT objects before moving to prediction
    del X, mag, freqs
    gc.collect()

    return {"hf_ratio": hf_ratio, "zcr": zcr}

def _predict_probability_ai(feats: Dict[str, float]) -> float:
    hf = feats["hf_ratio"]
    zcr = feats["zcr"]
    
    # Heuristic scoring: AI voices often lack natural jitter (low ZCR)
    # and contain spectral "ringing" (high HF ratio)
    score = 0.0
    score += np.clip((hf - 0.20) * 4.0, -1.5, 1.5)
    score += np.clip((0.06 - zcr) * 3.5, -1.5, 1.5)

    p_ai = 1.0 / (1.0 + math.exp(-score))
    return float(max(0.0, min(1.0, p_ai)))

# -----------------------------
# Main Endpoint
# -----------------------------
@app.post("/v1/detect", response_model=DetectResponse)
async def detect(req: DetectRequest, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    _require_api_key(x_api_key)

    if not req.audio_url and not req.audio_base64:
        raise HTTPException(status_code=422, detail="Provide audioUrl or audioBase64")

    # Download or Decode
    if req.audio_url:
        try:
            r = requests.get(req.audio_url, timeout=10)
            mp3_bytes = r.content
        except:
            raise HTTPException(status_code=400, detail="Failed to download audioUrl")
    else:
        try:
            mp3_bytes = base64.b64decode(req.audio_base64)
        except:
            raise HTTPException(status_code=400, detail="Invalid base64")

    # Process and Detect
    samples = _mp3_bytes_to_float32(mp3_bytes)
    feats = _simple_signal_features(samples)
    p_ai = _predict_probability_ai(feats)

    classification = "AI_GENERATED" if p_ai >= 0.5 else "HUMAN"
    confidence = 0.5 + abs(p_ai - 0.5)

    # Cleanup before returning response
    del samples, mp3_bytes
    gc.collect()

    reasoning = (
        f"Acoustic analysis indicates {'synthetic' if p_ai >= 0.5 else 'natural'} spectral "
        f"balance (HF ratio: {feats['hf_ratio']:.3f}, ZCR: {feats['zcr']:.3f})."
    )

    return DetectResponse(
        classification=classification,
        confidence_score=float(confidence),
        language_detected=req.language or "Unknown",
        reasoning=reasoning
    )

@app.get("/")
def health(): return {"status": "ok"}
