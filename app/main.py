"""API FastAPI : Secure GPU Upscaler."""
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.engine import UpscaleEngine
from app.metrics import compute_ssim
from app.security import MAX_FILE_SIZE, validate_image_upload

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "weights" / "RealESRGAN_x4plus_anime_6B.pth"))
TILE_SIZE = 512


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement unique du modèle en VRAM
    app.state.engine = UpscaleEngine(MODEL_PATH, device="cuda", half=True)
    app.state.gpu_lock = threading.Lock()  # une seule inférence GPU à la fois
    yield
    del app.state.engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Secure GPU Upscaler API", lifespan=lifespan)


def _run_inference(engine: UpscaleEngine, lock: threading.Lock, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with lock:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result_rgb = engine.upscale_image(rgb, tile_size=TILE_SIZE)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    ssim = compute_ssim(rgb, result_rgb)
    return result_rgb, elapsed_ms, peak_mb, ssim


@app.post("/api/v1/upscale")
async def upscale(request: Request, file: UploadFile = File(...)):
    # Lecture bornée : jamais plus de MAX_FILE_SIZE + 1 octets en mémoire
    data = await file.read(MAX_FILE_SIZE + 1)
    bgr = validate_image_upload(data)

    engine = request.app.state.engine
    try:
        result_rgb, elapsed_ms, peak_mb, ssim = await run_in_threadpool(
            _run_inference, engine, request.app.state.gpu_lock, bgr
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="VRAM insuffisante, réessayer plus tard.")

    ok, png = cv2.imencode(".png", cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(status_code=500, detail="Échec de l'encodage de la sortie.")

    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={
            "X-Inference-Time-Ms": f"{elapsed_ms:.1f}",
            "X-VRAM-Usage-MB": f"{peak_mb:.1f}",
            "X-SSIM-Score": f"{ssim:.2f}",
        },
    )


@app.get("/api/v1/health")
async def health():
    if not torch.cuda.is_available():
        return {"status": "degraded", "gpu_name": None, "vram_free_mb": 0.0}
    free_bytes, _total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_free_mb": round(free_bytes / (1024 ** 2), 1),
    }

