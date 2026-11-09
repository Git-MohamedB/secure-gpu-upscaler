"""Métriques de qualité d'image : PSNR et SSIM (scikit-image)."""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _align(img_original: np.ndarray, img_upscaled: np.ndarray) -> np.ndarray:
    """Redimensionne l'original (bicubique) aux dimensions de l'image upscalée."""
    h, w = img_upscaled.shape[:2]
    if img_original.shape[:2] == (h, w):
        return img_original
    return cv2.resize(img_original, (w, h), interpolation=cv2.INTER_CUBIC)


def compute_psnr(img_original: np.ndarray, img_upscaled: np.ndarray) -> float:
    ref = _align(img_original, img_upscaled)
    value = peak_signal_noise_ratio(ref, img_upscaled, data_range=255)
    return round(float(value), 2)


def compute_ssim(img_original: np.ndarray, img_upscaled: np.ndarray) -> float:
    ref = _align(img_original, img_upscaled)
    value = structural_similarity(ref, img_upscaled, channel_axis=2, data_range=255)
    return round(float(value), 2)

