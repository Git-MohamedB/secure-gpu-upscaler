"""Télécharge les poids Real-ESRGAN (anime 6B) dans weights/ s'ils sont absents."""
import sys
import urllib.request
from pathlib import Path

WEIGHTS_NAME = "RealESRGAN_x4plus_anime_6B.pth"
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    f"v0.2.2.4/{WEIGHTS_NAME}"
)
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def download_weights(dest_dir: Path = WEIGHTS_DIR) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / WEIGHTS_NAME
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] Poids déjà présents : {dest}")
        return dest

    tmp = dest.with_suffix(".pth.part")
    print(f"[..] Téléchargement de {WEIGHTS_URL}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r     {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp, _progress)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)  # écriture atomique : pas de fichier tronqué
    print(f"\n[OK] Poids enregistrés : {dest}")
    return dest


if __name__ == "__main__":
    try:
        download_weights()
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)

