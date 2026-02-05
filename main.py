# main.py
# Render-ready FastAPI API for AI-Generated Voice Detection
# Supports: Tamil, English, Hindi, Malayalam, Telugu
# Input: Base64 MP3
# Output: JSON {classification, confidence_score, language_detected, reasoning}

import os
import io
import base64
import math
from typing import Optional, Literal, Dict

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import numpy as np
from pydub import AudioSegment


# -----------------------------
# Config
# -----------------------------
API_KEY_ENV = "API_KEY"  # set this in Render Environment Variables

MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "20"))
TARGET_SR = 16000


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="AI-Generated Voice Detection API",
    version="1.0.0",
    description="Detect whether a Base64 MP3 voice sample is AI-generated or human.",
)


# -----------------------------
# Request/Response Schemas
# -----------------------------
SupportedLanguage = Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu", "Unknown"]


class DetectRequest(BaseModel):
    # GUVI tester sends camelCase keys
    audio_base64: str = Field(..., alias="audioBase64", description="Base64 encoded MP3 audio")

    language: Optional[SupportedLanguage] = Field(
        default=None,
        description="Optional language hint: Tamil/English/Hindi/Malayalam/Telugu",
    )

    audio_format: Optional[str] = Field(
        default="mp3",
        alias="audioFormat",
        description="Audio format (expected: mp3)",
    )

    audio_base64_format: Optional[str] = Field(
        default="base64",
        alias="audioBase64Format",
        description="Base64 format indicator (expected: base64)",
    )

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


def _decode_base64_mp3_to_wav_bytes(audio_b64: str) -> bytes:
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    try:
        audio = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode MP3. Ensure valid MP3 base64")

    # Force mono + 16kHz
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

    # Trim
    max_ms = int(MAX_AUDIO_SECONDS * 1000)
    if len(audio) > max_ms:
        audio = audio[:max_ms]

    wav_buf = io.BytesIO()
    audio.export(wav_buf, format="wav")
    return wav_buf.getvalue()


def _wav_bytes_to_float32(wav_bytes: bytes) -> np.ndarray:
    try:
        audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    except Exception:
        raise HTTPException(status_code=400, detail="WAV conversion failed")

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Normalize to [-1, 1]
    max_val = float(np.max(np.abs(samples)) + 1e-9)
    samples = samples / max_val
    return samples


def _language_detect(language_hint: Optional[str]) -> str:
    if language_hint in {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}:
        return language_hint
    return "Unknown"


def _safe_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _confidence_from_probability(p_ai: float) -> float:
    # 0.5 => 0.5 confidence, 0 or 1 => 1.0 confidence
    conf = 0.5 + abs(p_ai - 0.5)
    return float(max(0.0, min(1.0, conf)))


# -----------------------------
# Acoustic Feature Extraction (Lightweight)
# -----------------------------
def _simple_signal_features(x: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    frame = int(0.02 * sr)  # 20ms
    hop = int(0.01 * sr)    # 10ms

    if len(x) < frame:
        return {"hf_ratio": 0.0, "zcr": 0.0, "energy_var": 0.0}

    energies = []
    zcrs = []

    for i in range(0, len(x) - frame, hop):
        w = x[i:i + frame]

        energies.append(float(np.mean(w ** 2)))

        # ZCR
        zc = np.mean(np.abs(np.diff(np.sign(w))) > 0)
        zcrs.append(float(zc))

    energy_var = float(np.var(energies)) if energies else 0.0
    zcr = float(np.mean(zcrs)) if zcrs else 0.0

    # High-frequency ratio
    X = np.fft.rfft(x)
    mag = np.abs(X) + 1e-9
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)

    hf = mag[freqs >= 6000].sum()
    total = mag.sum()
    hf_ratio = float(hf / total)

    return {"hf_ratio": hf_ratio, "zcr": zcr, "energy_var": energy_var}


# -----------------------------
# Probability Scoring (No ML model to avoid OOM)
# -----------------------------
def _predict_ai_probability_from_feats(feats: Dict[str, float]) -> float:
    hf_ratio = feats["hf_ratio"]
    zcr = feats["zcr"]
    energy_var = feats["energy_var"]

    # Convert each feature into a bounded AI-likeness score [0,1]
    # (based on acoustic artifacts, not speech content)

    # HF artifacts: AI tends to be higher
    s_hf = np.clip((hf_ratio - 0.12) / (0.30 - 0.12), 0.0, 1.0)

    # ZCR: AI often has too-stable micro-jitter (lower)
    s_zcr = np.clip((0.09 - zcr) / (0.09 - 0.03), 0.0, 1.0)

    # Energy variance: AI often flatter dynamics (lower)
    s_env = np.clip((0.03 - energy_var) / (0.03 - 0.005), 0.0, 1.0)

    # Weighted sum
    score = 0.45 * s_hf + 0.30 * s_zcr + 0.25 * s_env
    score = float(np.clip(score, 0.0, 1.0))

    # Smooth it slightly
    # Map [0,1] -> logit -> sigmoid for stable curve
    logit = (score - 0.5) * 4.0
    p_ai = _safe_sigmoid(logit)

    return float(max(0.0, min(1.0, p_ai)))


def _explainability_signals(p_ai: float, hf_ratio: float, zcr: float, energy_var: float) -> str:
    verdict = "AI-like" if p_ai >= 0.5 else "Human-like"

    hf_txt = "elevated high-frequency artifacts" if hf_ratio > 0.22 else "natural high-frequency roll-off"
    zcr_txt = "over-stable micro-variations" if zcr < 0.06 else "organic micro-variations"
    env_txt = "flattened loudness dynamics" if energy_var < 0.015 else "natural loudness dynamics"

    return f"{verdict} acoustic profile: {hf_txt}, {zcr_txt}, {env_txt}."


# -----------------------------
# Error handling
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/")
def root():
    return {"status": "ok", "service": "ai-voice-detector", "endpoint": "/v1/detect"}


@app.post("/v1/detect", response_model=DetectResponse)
def detect(
    req: DetectRequest,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
):
    _require_api_key(x_api_key)

    # 1) Decode base64 MP3 -> WAV
    wav_bytes = _decode_base64_mp3_to_wav_bytes(req.audio_base64)

    # 2) WAV -> float32
    audio = _wav_bytes_to_float32(wav_bytes)

    # Minimum length
    if len(audio) < int(0.5 * TARGET_SR):
        raise HTTPException(status_code=400, detail="Audio too short. Provide at least 0.5 seconds")

    # 3) Features
    feats = _simple_signal_features(audio, TARGET_SR)

    # 4) Predict probability
    p_ai = _predict_ai_probability_from_feats(feats)

    classification = "AI_GENERATED" if p_ai >= 0.5 else "HUMAN"
    confidence = _confidence_from_probability(p_ai)

    # 5) Explainability
    reasoning = _explainability_signals(
        p_ai=p_ai,
        hf_ratio=feats["hf_ratio"],
        zcr=feats["zcr"],
        energy_var=feats["energy_var"],
    )

    # 6) Language
    language_detected = _language_detect(req.language)

    return DetectResponse(
        classification=classification,
        confidence_score=float(confidence),
        language_detected=language_detected,
        reasoning=reasoning,
    )
