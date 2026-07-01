#!/usr/bin/env bash
# =============================================================================
# start_application.sh – CodeDeploy ApplicationStart hook
# Builds Docker images from the newly deployed revision and starts the full
# Docker Compose stack.  Performs a health-check loop to confirm the backend
# is accepting requests before marking the deployment as successful.
# =============================================================================
set -euo pipefail

APP_DIR="/opt/entrapeer"
LOG_FILE="/var/log/codedeploy-entrapeer.log"
HEALTH_URL="http://localhost:8000/health"
HEALTH_RETRIES=20
HEALTH_INTERVAL=5  # seconds

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== ApplicationStart: starting ==="

# ── 1. Sanity check ───────────────────────────────────────────────────────────
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
    log "ERROR: docker-compose.yml not found in $APP_DIR"
    exit 1
fi

cd "$APP_DIR"

# ── 2. Validate .env exists (do not create one – must be pre-provisioned) ─────
if [ ! -f ".env" ]; then
    log "ERROR: .env file not found. Provision it on the server before deploying."
    exit 1
fi

# ── 3. Build images and start all services ───────────────────────────────────
log "Building Docker images..."
docker compose build --no-cache

log "Starting Docker Compose stack (detached)..."
docker compose up -d

log "Stack started. Running containers:"
docker compose ps | tee -a "$LOG_FILE"

# ── 4. Health-check loop ──────────────────────────────────────────────────────
log "Waiting for backend to become healthy (${HEALTH_URL})..."
for i in $(seq 1 "$HEALTH_RETRIES"); do
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
    if [ "$HTTP_STATUS" = "200" ]; then
        log "Backend is healthy (attempt $i/$HEALTH_RETRIES) – HTTP $HTTP_STATUS"
        break
    fi
    log "Attempt $i/$HEALTH_RETRIES – HTTP $HTTP_STATUS – retrying in ${HEALTH_INTERVAL}s..."
    sleep "$HEALTH_INTERVAL"

    if [ "$i" -eq "$HEALTH_RETRIES" ]; then
        log "ERROR: Backend did not become healthy after $((HEALTH_RETRIES * HEALTH_INTERVAL))s"
        log "--- Last 50 lines of web container logs ---"
        docker compose logs --tail=50 web | tee -a "$LOG_FILE"
        exit 1
    fi
done

log "=== ApplicationStart: complete — deployment successful ==="
