"""Tests d'integration API FastAPI executes sur CPU avec mock CUDA/moteur."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.main as main_mod
from app.security import MAX_FILE_SIZE


class _FakeUpscaleEngine:
    def __init__(self, model_path: str, device: str = "cuda", half: bool = True):
        self.model_path = model_path
        self.device = device
        self.half = half
        self.raise_oom = False

    def upscale_image(self, image_np: np.ndarray, tile_size: int = 512) -> np.ndarray:
        if self.raise_oom:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (mock)")
        h, w = image_np.shape[:2]
        return cv2.resize(image_np, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


def _png_bytes(w: int = 32, h: int = 32) -> bytes:
    img = np.random.default_rng(7).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def cpu_client(monkeypatch):
    empty_cache_mock = MagicMock()
    monkeypatch.setattr(main_mod, "UpscaleEngine", _FakeUpscaleEngine)
    monkeypatch.setattr(main_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        main_mod.torch.cuda, "mem_get_info", lambda: (4096 * 1024 * 1024, 8192 * 1024 * 1024)
    )
    monkeypatch.setattr(main_mod.torch.cuda, "get_device_name", lambda _idx=0: "Mock CUDA GPU")
    monkeypatch.setattr(main_mod.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(main_mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(main_mod.torch.cuda, "max_memory_allocated", lambda: 128 * 1024 * 1024)
    monkeypatch.setattr(main_mod.torch.cuda, "empty_cache", empty_cache_mock)

    with TestClient(main_mod.app) as client:
        yield client, empty_cache_mock


def test_health_mocked_cuda(cpu_client):
    client, _ = cpu_client
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["gpu_name"] == "Mock CUDA GPU"
    assert body["vram_free_mb"] == 4096.0


def test_upscale_returns_4x_and_request_id_headers(cpu_client):
    client, _ = cpu_client
    payload = _png_bytes(32, 24)

    r_echo = client.post(
        "/api/v1/upscale",
        headers={"X-Request-ID": "req-custom-42"},
        files={"file": ("sample.png", payload, "image/png")},
    )
    assert r_echo.status_code == 200
    assert r_echo.headers["X-Request-ID"] == "req-custom-42"
    for h in ("X-Inference-Time-Ms", "X-VRAM-Usage-MB", "X-SSIM-Score"):
        assert h in r_echo.headers

    out = cv2.imdecode(np.frombuffer(r_echo.content, np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (96, 128, 3)

    r_gen = client.post(
        "/api/v1/upscale",
        files={"file": ("sample.png", payload, "image/png")},
    )
    assert r_gen.status_code == 200
    assert len(r_gen.headers["X-Request-ID"]) == 12


def test_upscale_rejects_fake_png_magic_bytes(cpu_client):
    client, _ = cpu_client
    r = client.post(
        "/api/v1/upscale",
        files={"file": ("fake.png", b"not-a-real-png-header", "image/png")},
    )
    assert r.status_code == 415


def test_upscale_rejects_oversized_upload(cpu_client):
    client, _ = cpu_client
    oversized = b"\x89PNG\r\n\x1a\n" + (b"0" * MAX_FILE_SIZE)
    r = client.post(
        "/api/v1/upscale",
        files={"file": ("big.png", oversized, "image/png")},
    )
    assert r.status_code == 413


def test_upscale_returns_503_and_clears_cache_on_oom(cpu_client):
    client, empty_cache_mock = cpu_client
    main_mod.app.state.engine.raise_oom = True

    r = client.post(
        "/api/v1/upscale",
        files={"file": ("sample.png", _png_bytes(16, 16), "image/png")},
    )
    assert r.status_code == 503
    assert empty_cache_mock.call_count >= 1
