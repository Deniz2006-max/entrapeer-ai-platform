#!/usr/bin/env bash
# =============================================================================
# before_install.sh – CodeDeploy BeforeInstall hook
# Runs on the EC2 instance BEFORE the new revision is copied.
# Responsibilities:
#   • Ensure Docker & Docker Compose are installed and running.
#   • Pull latest base images to warm up the layer cache.
#   • Create the application directory with correct permissions.
# =============================================================================
set -euo pipefail

APP_DIR="/opt/entrapeer"
LOG_FILE="/var/log/codedeploy-entrapeer.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== BeforeInstall: starting ==="

# ── 1. Ensure Docker is installed ────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Docker not found – installing..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    log "Docker installed: $(docker --version)"
else
    log "Docker already installed: $(docker --version)"
fi

# ── 2. Ensure Docker Compose plugin is available ─────────────────────────────
if ! docker compose version &>/dev/null; then
    log "Docker Compose plugin not found – installing..."
    COMPOSE_VERSION="v2.27.0"
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    log "Docker Compose installed: $(docker compose version)"
else
    log "Docker Compose already installed: $(docker compose version)"
fi

# ── 3. Ensure Docker daemon is running ───────────────────────────────────────
if ! systemctl is-active --quiet docker; then
    log "Starting Docker daemon..."
    systemctl start docker
fi

# ── 4. Create application directory ──────────────────────────────────────────
mkdir -p "$APP_DIR"
chmod 755 "$APP_DIR"
log "Application directory ready: $APP_DIR"

# ── 5. Pre-pull heavy base images to speed up build ──────────────────────────
log "Pre-pulling base images..."
docker pull python:3.11-slim  || log "WARN: python:3.11-slim pull failed (non-fatal)"
docker pull node:22-alpine    || log "WARN: node:22-alpine pull failed (non-fatal)"
docker pull mongo:8           || log "WARN: mongo:8 pull failed (non-fatal)"
docker pull redis:8-alpine    || log "WARN: redis:8-alpine pull failed (non-fatal)"

log "=== BeforeInstall: complete ==="
