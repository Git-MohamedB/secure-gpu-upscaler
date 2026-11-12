# PROJECT_STATE.md

## 1. Contexte & Contraintes
- **Projet :** Secure GPU-Accelerated Microservice (Super-Resolution & DoS Defense)
- **Modèle :** Real-ESRGAN x4plus anime 6B (RRDBNet)
- **Matériel cible :** GPU 8 Go VRAM (NVIDIA GeForce RTX 4060 Ti), FP16 Half Precision, Tiling spatial dynamique
- **OS :** Linux / WSL2 Ubuntu 22.04
- **Sécurité :** Conteneur Docker Rootless (UID 10001 `appuser`), quotas cgroups (4 Go RAM, 2 CPU), no-new-privileges, validation stricte Magic Bytes et protection Decompression Bomb.

## 2. Stack & Versions
- **Runtime :** Python 3.10 / PyTorch 2.3.1 CUDA 12.1 / CUDA 12.2 Runtime
- **API :** FastAPI 0.111.0 / Uvicorn 0.30.1 / Starlette
- **Traitement d'image & Métriques :** OpenCV 4.10 / Pillow 10.4 / scikit-image 0.24
- **Conteneurisation & Durcissement :** Docker 29.x / Docker Compose v2.x / cgroups v2

## 3. Avancement
- [x] **Jalon 0 :** Arborescence et environnement initialisés (structure modulaire, gitignore, requirements)
- [x] **Jalon 1 :** Moteur GPU local, FP16 et Métriques (RRDBNet standalone, weights_only=True, tiling 512x512 avec 10px pad, PSNR & SSIM)
- [x] **Jalon 2 :** API FastAPI et Sécurité anti-DoS (rejet 413 >10 Mo, 415 magic bytes, 400 decompression bomb >2048x2048, lifespan VRAM, headers métriques, suite pytest)
- [x] **Jalon 3 :** Dockerisation GPU et Durcissement (CUDA 12.2 runtime, non-root appuser UID 10001, cgroups 4g/2.0 CPU, no-new-privileges, passthrough GPU NVIDIA)
- [x] **Jalon 4 :** Audit de sécurité et Documentation CV (script d'audit `scripts/security_audit.sh`, README technique complet niveau ingénieur avec schémas, formules mathématiques et benchmarks réels)

## 4. Bilan & Validation Finale
- **Statut global :** **TERMINÉ (100 % des jalons validés)**.
- **Résultats physiques sur RTX 4060 Ti :**
  - VRAM pic bornée : **~559 - 562 Mo** (invariance garantie par le tuilage spatial, immunité totale aux OOM GPU).
  - Latence : **~45 ms** (64x64), **~326 ms** (256x256), **~680 ms** (512x512).
  - Télémétrie HTTP : en-têtes `X-Inference-Time-Ms`, `X-VRAM-Usage-MB`, `X-SSIM-Score` injectés dynamiquement.
  - Audit de conformité : 0 fail sur l'inspection conteneur, exécution non-root validée (`uid=10001(appuser)`).
