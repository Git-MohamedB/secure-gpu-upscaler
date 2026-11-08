"""Tests Jalon 2 : API FastAPI et sécurité applicative (nécessite un GPU CUDA)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.security import validate_image_upload


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # déclenche le lifespan (chargement du modèle)
        yield c


def _png_bytes(w=64, h=64) -> bytes:
    img = np.random.default_rng(0).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["gpu_name"]
    assert body["vram_free_mb"] > 0


def test_reject_text_file_renamed_png(client):
    r = client.post(
        "/api/v1/upscale",
        files={"file": ("fake.png", b"ceci n'est pas une image", "image/png")},
    )
    assert r.status_code == 415


def test_nominal_upload(client):
    r = client.post("/api/v1/upscale", files={"file": ("ok.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    for h in ("X-Inference-Time-Ms", "X-VRAM-Usage-MB", "X-SSIM-Score"):
        assert h in r.headers
    out = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (256, 256, 3)


def test_reject_oversized_file():
    with pytest.raises(HTTPException) as e:
        validate_image_upload(b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024))
    assert e.value.status_code == 413


def test_reject_oversized_dimensions():
    with pytest.raises(HTTPException) as e:
        validate_image_upload(_png_bytes(2049, 16))
    assert e.value.status_code == 400

