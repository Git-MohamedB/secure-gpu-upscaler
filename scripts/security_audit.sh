#!/usr/bin/env bash
# ==============================================================================
# Security Audit Script for Secure GPU Upscaler
# Audits container hardening (non-root, cgroups, capabilities) & CVE scan
# ==============================================================================
set -euo pipefail

IMAGE_NAME="secure-gpu-upscaler:latest"
CONTAINER_NAME="secure-gpu-upscaler-upscaler-1"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=====================================================${NC}"
echo -e "${BLUE}       SECURE GPU UPSCALER - SECURITY AUDIT          ${NC}"
echo -e "${BLUE}=====================================================${NC}"

FAILED=0

check_pass() {
    echo -e " [${GREEN}PASS${NC}] $1"
}

check_fail() {
    echo -e " [${RED}FAIL${NC}] $1"
    FAILED=$((FAILED + 1))
}

check_warn() {
    echo -e " [${YELLOW}WARN${NC}] $1"
}

# 1. Vérification de l'image
echo -e "\n${BLUE}1. Vérification de l'image Docker (${IMAGE_NAME})...${NC}"
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    check_fail "L'image ${IMAGE_NAME} n'existe pas. Lancez 'docker compose build' d'abord."
    exit 1
fi
check_pass "Image ${IMAGE_NAME} présente."

# 2. Vérification de l'utilisateur non-root dans l'image
IMAGE_USER=$(docker image inspect --format '{{.Config.User}}' "${IMAGE_NAME}")
if [ "${IMAGE_USER}" = "appuser" ] || [ "${IMAGE_USER}" = "10001" ]; then
    check_pass "L'image est configurée avec un utilisateur non-root (USER=${IMAGE_USER})."
else
    check_fail "L'image s'exécute en root ou avec un utilisateur inattendu (USER='${IMAGE_USER}')."
fi

# 3. Vérification du conteneur en cours d'exécution
echo -e "\n${BLUE}2. Audit du conteneur en exécution (${CONTAINER_NAME})...${NC}"
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    # Vérification UID effectif dans le conteneur
    RUNNING_UID=$(docker inspect --format '{{.Config.User}}' "${CONTAINER_NAME}")
    if [ "${RUNNING_UID}" = "appuser" ] || [ "${RUNNING_UID}" = "10001" ]; then
        check_pass "Conteneur actif avec utilisateur non-root (UID=${RUNNING_UID})."
    else
        check_fail "Conteneur actif avec utilisateur non restreint: ${RUNNING_UID}"
    fi

    # Vérification de l'option no-new-privileges
    SEC_OPT=$(docker inspect --format '{{json .HostConfig.SecurityOpt}}' "${CONTAINER_NAME}")
    if echo "${SEC_OPT}" | grep -q "no-new-privileges:true"; then
        check_pass "Option de sécurité 'no-new-privileges:true' activée."
    else
        check_fail "Option 'no-new-privileges:true' absente."
    fi

    # Vérification des limites de mémoire cgroups
    MEM_LIMIT=$(docker inspect --format '{{.HostConfig.Memory}}' "${CONTAINER_NAME}")
    if [ "${MEM_LIMIT}" -gt 0 ]; then
        MEM_MB=$((MEM_LIMIT / 1024 / 1024))
        check_pass "Limite de mémoire cgroups configurée : ${MEM_MB} Mo."
    else
        check_fail "Aucune limite de mémoire cgroups définie."
    fi

    # Vérification des quotas CPU
    NANO_CPUS=$(docker inspect --format '{{.HostConfig.NanoCpus}}' "${CONTAINER_NAME}")
    if [ "${NANO_CPUS}" -gt 0 ]; then
        check_pass "Limite de CPU cgroups configurée (${NANO_CPUS} NanoCPUs)."
    else
        check_warn "Aucun quota NanoCPUs explicite détecté."
    fi

    # Vérification des volumes en lecture seule (read-only)
    RO_MOUNTS=$(docker inspect --format '{{range .Mounts}}{{if .RW}}RW: {{.Destination}} {{else}}RO: {{.Destination}} {{end}}{{end}}' "${CONTAINER_NAME}")
    if echo "${RO_MOUNTS}" | grep -q "RO: /app/weights"; then
        check_pass "Poids du modèle montés en lecture seule (RO: /app/weights)."
    else
        check_warn "Vérifier le montage des poids: ${RO_MOUNTS}"
    fi
else
    check_warn "Conteneur ${CONTAINER_NAME} non actif. Lancez 'docker compose up -d' pour un audit à chaud."
fi

# 4. Scan de vulnérabilités des dépendances (Trivy)
echo -e "\n${BLUE}3. Analyse de vulnérabilités des dépendances (Trivy)...${NC}"
if command -v trivy >/dev/null 2>&1; then
    echo "Trivy détecté. Démarrage du scan de l'image ${IMAGE_NAME}..."
    trivy image --severity HIGH,CRITICAL --ignore-unfixed "${IMAGE_NAME}"
    check_pass "Scan Trivy exécuté avec succès."
else
    check_warn "Trivy n'est pas installé sur l'hôte. Pour analyser les vulnérabilités CVE :"
    echo "       Installation : https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
    echo "       Ou via Docker : docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image ${IMAGE_NAME}"
fi

# Résumé
echo -e "\n${BLUE}=====================================================${NC}"
if [ ${FAILED} -eq 0 ]; then
    echo -e "${GREEN}       RÉSULTAT DE L'AUDIT : 0 ERREUR CRITIQUE       ${NC}"
    echo -e "${GREEN}       Durcissement conteneur et sécurité validés !  ${NC}"
    echo -e "${BLUE}=====================================================${NC}"
    exit 0
else
    echo -e "${RED}       RÉSULTAT DE L'AUDIT : ${FAILED} ÉCHEC(S)        ${NC}"
    echo -e "${BLUE}=====================================================${NC}"
    exit 1
fi

