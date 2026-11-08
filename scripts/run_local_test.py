"""Test local Jalon 1 : upscale x4 d'une matrice aléatoire 256x256 sur GPU."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from app.engine import UpscaleEngine
from app.metrics import compute_psnr, compute_ssim

MODEL_PATH = ROOT / "weights" / "RealESRGAN_x4plus_anime_6B.pth"
OUTPUT_PATH = ROOT / "output" / "test_result.png"


def main() -> int:
    if not torch.cuda.is_available():
        print("[ERREUR] CUDA indisponible : PyTorch avec support GPU requis.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    engine = UpscaleEngine(str(MODEL_PATH), device="cuda", half=True)

    # Mesure du pic VRAM : reset après chargement du modèle, puis inférence
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = engine.upscale_image(image, tile_size=512)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    assert result.shape == (1024, 1024, 3), f"Taille inattendue : {result.shape}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Sortie         : {result.shape[1]}x{result.shape[0]} -> {OUTPUT_PATH}")
    print(f"Temps          : {elapsed_ms:.1f} ms")
    print(f"VRAM pic       : {peak_mb:.1f} Mo")
    print(f"PSNR           : {compute_psnr(image, result)} dB")
    print(f"SSIM           : {compute_ssim(image, result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

