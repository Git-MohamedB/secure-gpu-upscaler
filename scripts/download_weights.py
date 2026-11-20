"""Télécharge les poids Real-ESRGAN (anime 6B) dans weights/ s'ils sont absents."""
import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS_NAME = "RealESRGAN_x4plus_anime_6B.pth"
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    f"v0.2.2.4/{WEIGHTS_NAME}"
)
WEIGHTS_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def compute_sha256(file_path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier par blocs de 1 Mio."""
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(file_path: Path, expected_sha256: str = WEIGHTS_SHA256) -> None:
    """Verifie que l'empreinte SHA-256 correspond au condensat attendu."""
    actual = compute_sha256(file_path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Empreinte SHA-256 invalide pour {file_path.name} : "
            f"attendu {expected_sha256}, obtenu {actual}"
        )


def download_weights(
    dest_dir: Path = WEIGHTS_DIR,
    expected_sha256: str = WEIGHTS_SHA256,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / WEIGHTS_NAME
    if dest.exists() and dest.stat().st_size > 0:
        verify_sha256(dest, expected_sha256)
        print(f"[OK] Poids déjà présents et vérifiés (SHA-256) : {dest}")
        return dest

    tmp = dest.with_suffix(".pth.part")
    print(f"[..] Téléchargement de {WEIGHTS_URL}")

    def _progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r     {pct:3d}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp, _progress)
        verify_sha256(tmp, expected_sha256)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)  # écriture atomique : pas de fichier tronqué
    print(f"\n[OK] Poids enregistrés et vérifiés (SHA-256) : {dest}")
    return dest


if __name__ == "__main__":
    try:
        download_weights()
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)

