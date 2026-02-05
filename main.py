# main.py
# GUVI + Render-ready FastAPI API for AI-Generated Voice Detection
# Supports: Tamil, English, Hindi, Malayalam, Telugu
# Input: audioUrl (GUVI tester) OR audioBase64
# Output: JSON {classification, confidence_score, language_detected, reasoning}

import os
import io
import base64
import math
from typing import Optional, Literal, Dict

import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pydub import AudioSegment


# -----------------------------
# Config
# -----------------------------
API_KEY_ENV = "API_KEY"
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "20"))
TARGET_SR = 16000


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="AI-Generated Voice Detection API",
    version="1.0.0",
    description="Detect whether a voice sample is AI-generated or human.",
)


# -----------------------------
# Request/Response Schemas
# -----------------------------
SupportedLanguage = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu", "Unknown"]


class DetectRequest(BaseModel):
    # GUVI tester might send audioUrl
    audio_url: Optional[str] = Field(default=None, alias="audioUrl")

    # Your base64 support
    audio_base64: Optional[str] = Field(default=None, alias="audioBase64")

    language: Optional[SupportedLanguage] = Field(default=None)

    audio_format: Optional[str] = Field(default="mp3", alias="audioFormat")
    audio_base64_format: Optional[str] = Field(default="base64", alias="audioBase64Format")

    model_config = {"populate_by_name": True}


class DetectResponse(BaseModel):
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    language_detected: SupportedLanguage
    reasoning: str


# -----------------------------
# Helpers
# -----------------------------
def _require_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv(API_KEY_ENV)
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _download_audio(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download audio from audioUrl")
        return r.content
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to download audio from audioUrl")


def _decode_base64(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")


def _mp3_bytes_to_float32(mp3_bytes: bytes) -> np.ndarray:
    # Convert MP3 -> WAV -> float32
    try:
        audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode MP3")

    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

    # Trim for stability
    max_ms = int(MAX_AUDIO_SECONDS * 1000)
    if len(audio) > max_ms:
        audio = audio[:max_ms]

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize [-1, 1]
    peak = float(np.max(np.abs(samples)) + 1e-9)
    samples = samples / peak

    return samples


def _language_detect(language_hint: Optional[str]) -> str:
    if language_hint in {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}:
        return language_hint
    return "Unknown"


def _simple_signal_features(x: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    # lightweight acoustic features
    frame = int(0.02 * sr)  # 20ms
    hop = int(0.01 * sr)    # 10ms

    if len(x) < frame:
        return {"hf_ratio": 0.0, "zcr": 0.0, "energy_var": 0.0}

    energies = []
    zcrs = []
    for i in range(0, len(x) - frame, hop):
        w = x[i:i + frame]
        energies.append(float(np.mean(w ** 2)))
        zc = np.mean(np.abs(np.diff(np.sign(w))) > 0)
        zcrs.append(float(zc))

    energy_var = float(np.var(energies)) if energies else 0.0
    zcr = float(np.mean(zcrs)) if zcrs else 0.0

    X = np.fft.rfft(x)
    mag = np.abs(X) + 1e-9
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)

    hf = mag[freqs >= 6000].sum()
    total = mag.sum()
    hf_ratio = float(hf / total)

    return {"hf_ratio": hf_ratio, "zcr": zcr, "energy_var": energy_var}


def _predict_probability_ai(feats: Dict[str, float]) -> float:
    # Simple deterministic scoring (NO hardcoding speech content)
    hf = feats["hf_ratio"]
    zcr = feats["zcr"]
    env = feats["energy_var"]

    # Normalize signals into a score
    score = 0.0

    # AI voices often have slightly unnatural HF patterns
    score += np.clip((hf - 0.18) * 3.0, -1.0, 1.0)

    # AI voices often have more stable micro-variations
    score += np.clip((0.07 - zcr) * 3.0, -1.0, 1.0)

    # AI voices often have flatter energy dynamics
    score += np.clip((0.02 - env) * 15.0, -1.0, 1.0)

    # Convert score -> probability
    p_ai = 1.0 / (1.0 + math.exp(-score))
    return float(max(0.0, min(1.0, p_ai)))


def _confidence_from_probability(p_ai: float) -> float:
    conf = 0.5 + abs(p_ai - 0.5)
    return float(max(0.0, min(1.0, conf)))


def _explainability(p_ai: float, feats: Dict[str, float]) -> str:
    verdict = "AI-like" if p_ai >= 0.5 else "Human-like"

    hf_txt = "elevated high-frequency artifacts" if feats["hf_ratio"] > 0.22 else "natural high-frequency roll-off"
    zcr_txt = "over-stable micro-variations" if feats["zcr"] < 0.06 else "organic micro-variations"
    env_txt = "flattened loudness dynamics" if feats["energy_var"] < 0.015 else "natural loudness dynamics"

    return f"{verdict} acoustic profile: {hf_txt}, {zcr_txt}, {env_txt}."


# -----------------------------
# Error handling
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "/v1/detect"}


# -----------------------------
# Main Endpoint
# -----------------------------
@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    _require_api_key(x_api_key)

    # Must provide either audioUrl or audioBase64
    if not req.audio_url and not req.audio_base64:
        raise HTTPException(status_code=422, detail="Provide either audioUrl or audioBase64")

    # Get MP3 bytes
    if req.audio_url:
        mp3_bytes = _download_audio(req.audio_url)
    else:
        mp3_bytes = _decode_base64(req.audio_base64)

    # Convert to float32 samples
    audio = _mp3_bytes_to_float32(mp3_bytes)

    if len(audio) < int(0.5 * TARGET_SR):
        raise HTTPException(status_code=400, detail="Audio too short. Provide at least 0.5 seconds")

    feats = _simple_signal_features(audio, TARGET_SR)

    p_ai = _predict_probability_ai(feats)

    classification = "AI_GENERATED" if p_ai >= 0.5 else "HUMAN"
    confidence = _confidence_from_probability(p_ai)

    return DetectResponse(
        classification=classification,
        confidence_score=float(confidence),
        language_detected=_language_detect(req.language),
        reasoning=_explainability(p_ai, feats),
    )
