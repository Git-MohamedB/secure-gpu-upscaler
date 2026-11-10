"""Sécurité applicative : validation stricte des uploads d'images (anti-DoS)."""
import io

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 Mo
MAX_DIMENSION = 2048  # px, largeur et hauteur

# Pillow lève DecompressionBombError au-delà de ce nombre de pixels (garde-fou additionnel)
Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION


def _detect_format(data: bytes) -> str | None:
    """Identifie le format via les magic bytes (jamais via l'extension)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def validate_image_upload(file_bytes: bytes) -> np.ndarray:
    """Valide un upload et retourne l'image décodée (matrice OpenCV BGR uint8).

    - 413 : taille > 10 Mo
    - 415 : format autre que PNG / JPEG / WebP (magic bytes)
    - 400 : dimensions > 2048x2048, image corrompue ou illisible
    """
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 10 Mo).")

    fmt = _detect_format(file_bytes)
    if fmt is None:
        raise HTTPException(
            status_code=415, detail="Format non supporté (PNG, JPEG ou WebP uniquement)."
        )

    # Lecture des dimensions depuis l'en-tête uniquement (Image.open est paresseux :
    # aucun pixel n'est décompressé tant que load() n'est pas appelé).
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            if img.format != fmt:
                raise HTTPException(status_code=415, detail="Contenu incohérent avec le format.")
            width, height = img.size
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise HTTPException(status_code=400, detail="Image invalide ou corrompue.")

    if width < 1 or height < 1 or width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Dimensions interdites ({width}x{height}) : max {MAX_DIMENSION}x{MAX_DIMENSION}.",
        )

    # Décodage uniquement après validation des dimensions
    decoded = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Image invalide ou corrompue.")
    return decoded

