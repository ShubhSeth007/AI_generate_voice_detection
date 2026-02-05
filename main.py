import os
import io
import base64
import math
import gc 
from typing import Optional, Literal, Dict

import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pydub import AudioSegment

# -----------------------------
# Config - Optimized for 512MB RAM
# -----------------------------
API_KEY_ENV = "API_KEY"
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
    language: Optional[SupportedLanguage] = Field(default="Unknown")
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

def _safe_decode_base64(b64_str: str) -> bytes:
    try:
        b64_str = b64_str.strip()
        
        # SAFETY VALVE: Catch the "base64" placeholder string from the tester
        if b64_str.lower() == "base64" or len(b64_str) < 10:
             print("DEBUG: Placeholder string detected instead of real data.")
             # Return a tiny silent MP3 header or raise a clear error
             raise ValueError("Please provide a real Base64 audio string, not the word 'base64'.")

        if "," in b64_str:
            b64_str = b64_str.split(",")[-1]
        
        missing_padding = len(b64_str) % 4
        if missing_padding:
            b64_str += "=" * (4 - missing_padding)
            
        return base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _mp3_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    try:
        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError(f"Byte stream too small: {len(audio_bytes)} bytes")

        # MAGIC NUMBER CHECK: Inspect the first few bytes
        header = audio_bytes[:4]
        print(f"DEBUG: Byte Header (hex): {header.hex()}")

        # Determine format manually to help FFmpeg
        # RIFF = WAV, ID3/0xFFFB = MP3
        fmt = None
        if header.startswith(b'RIFF'):
            fmt = "wav"
        elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3'):
            fmt = "mp3"
        elif header.startswith(b'\x66\x74\x79\x70'): # ftyp
            fmt = "m4a"

        # Load with explicit format if found, otherwise auto-detect
        try:
            if fmt:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            else:
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        except Exception as inner_e:
            print(f"DEBUG: FFmpeg Error detail: {str(inner_e)}")
            raise ValueError(f"FFmpeg couldn't read {fmt if fmt else 'unknown'} format.")

        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
        max_ms = int(MAX_AUDIO_SECONDS * 1000)
        audio = audio[:max_ms]

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Free memory immediately
        del audio
        gc.collect()

        peak = float(np.max(np.abs(samples)) + 1e-9)
        return samples / peak

    except Exception as e:
        print(f"CRITICAL DECODE ERROR: {str(e)}")
        # We return the actual error message to the tester so you can see it in the response
        raise HTTPException(status_code=400, detail=f"Decoding Error: {str(e)}")

def _simple_signal_features(x: np.ndarray) -> Dict[str, float]:
    n = len(x)
    if n < 320: return {"hf_ratio": 0.0, "zcr": 0.0}

    X = np.fft.rfft(x)
    mag = np.abs(X) + 1e-9
    freqs = np.fft.rfftfreq(n, d=1.0 / TARGET_SR)

    hf = mag[freqs >= 6000].sum()
    hf_ratio = float(hf / (mag.sum() + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

    del X, mag, freqs
    gc.collect()

    return {"hf_ratio": hf_ratio, "zcr": zcr}

def _predict_probability_ai(feats: Dict[str, float]) -> float:
    hf = feats["hf_ratio"]
    zcr = feats["zcr"]
    
    score = 0.0
    score += np.clip((hf - 0.20) * 4.0, -1.5, 1.5)
    score += np.clip((0.06 - zcr) * 3.5, -1.5, 1.5)

    p_ai = 1.0 / (1.0 + math.exp(-score))
    return float(max(0.0, min(1.0, p_ai)))

# -----------------------------
# Main Endpoint
# -----------------------------
@app.post("/v1/detect", response_model=DetectResponse)
async def detect(request: Request, x_api_key: Optional[str] = Header(default=None, alias="x-api-key")):
    # 1. Manual Auth Check
    _require_api_key(x_api_key)

    # 2. Get the raw JSON body to see what the tester is REALLY sending
    body = await request.json()
    print(f"DEBUG: Full Request Body Keys: {list(body.keys())}")

    # 3. Greedy Extraction (Checks every possible variation of the key)
    audio_url = body.get("audioUrl") or body.get("audio_url")
    audio_b64 = body.get("audioBase64") or body.get("audio_base64") or body.get("audio")
    lang = body.get("language") or "Unknown"

    if not audio_url and not audio_b64:
        raise HTTPException(status_code=422, detail=f"No audio found. Keys received: {list(body.keys())}")

    # 4. Process
    if audio_url:
        try:
            mp3_bytes = requests.get(audio_url, timeout=10).content
        except:
            raise HTTPException(status_code=400, detail="Failed to download from audioUrl")
    else:
        # Use our safe decoder
        mp3_bytes = _safe_decode_base64(audio_b64)

    print(f"DEBUG: Final byte count for decoding: {len(mp3_bytes)}")

    # 5. Decode and Analyze
    samples = _mp3_bytes_to_float32(mp3_bytes)
    feats = _simple_signal_features(samples)
    p_ai = _predict_probability_ai(feats)

    classification = "AI_GENERATED" if p_ai >= 0.5 else "HUMAN"
    confidence = 0.5 + abs(p_ai - 0.5)

    return DetectResponse(
        classification=classification,
        confidence_score=float(confidence),
        language_detected=lang,
        reasoning=f"Analysis complete. Score: {p_ai:.2f}"
    )

@app.get("/")
def health(): return {"status": "ok"}





