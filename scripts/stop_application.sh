#!/usr/bin/env bash
# =============================================================================
# stop_application.sh – CodeDeploy ApplicationStop hook
# Gracefully stops the running Docker Compose stack before the new revision
# is deployed.  Missing containers are not treated as an error.
# =============================================================================
set -euo pipefail

APP_DIR="/opt/entrapeer"
LOG_FILE="/var/log/codedeploy-entrapeer.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== ApplicationStop: starting ==="

if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
    log "No docker-compose.yml found in $APP_DIR – nothing to stop."
    exit 0
fi

cd "$APP_DIR"

# ── Graceful stop with a 30-second timeout per container ─────────────────────
log "Stopping Docker Compose stack..."
docker compose down --timeout 30 --remove-orphans || {
    log "WARN: docker compose down reported an error – forcing removal..."
    docker compose kill 2>/dev/null || true
    docker compose rm -f 2>/dev/null || true
}

# ── Remove dangling images to reclaim disk space ─────────────────────────────
log "Pruning dangling Docker images..."
docker image prune -f || true

log "=== ApplicationStop: complete ==="
