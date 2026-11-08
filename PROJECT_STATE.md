# PROJECT_STATE.md
## 1. Contexte & Contraintes
- Projet : Secure GPU-Accelerated Microservice
- Matériel cible : GPU 8 Go VRAM, FP16, Tiling activé
- OS : Linux / WSL2 Ubuntu

## 2. Stack & Versions
- Python 3.10 / PyTorch CUDA 12.x / FastAPI / Docker Rootless

## 3. Avancement
- [x] Jalon 0 : Arborescence et environnement initialisés
- [x] Jalon 1 : Moteur GPU local, FP16 et Métriques
- [x] Jalon 2 : API FastAPI et Sécurité anti-DoS
- [ ] Jalon 3 : Dockerisation GPU et Durcissement
- [ ] Jalon 4 : Audit de sécurité et Documentation CV

## 4. Prochaine action immédiate
- Jalon 2 validé : 5 tests pytest OK (health, rejet faux PNG 415, upload nominal, 413, dimensions 400).
- Exécuter le Jalon 3 (Dockerisation GPU et durcissement).
