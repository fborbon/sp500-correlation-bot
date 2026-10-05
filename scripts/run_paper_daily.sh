#!/bin/bash
# Paper mode run — called by cron weekly (Mondays, matching the validated 7-trading-day
# rebalance cadence; TOP_N_POSITIONS selection and PREDICTION_DAYS were both tuned/tested
# at this frequency — running it daily would rebalance 5x more often than validated).
# Filename kept as run_paper_daily.sh to avoid changing the crontab path on the server.
# Runs inside the dashboard container so it shares the same cache/ and outputs/.

set -e
APP_DIR="/opt/forwardforecasting"
LOG="$APP_DIR/logs/paper_$(date +%Y-%m-%d).log"

mkdir -p "$APP_DIR/logs"

echo "=== $(date -u) — Starting paper run ===" >> "$LOG"

cd "$APP_DIR"
docker compose exec -T dashboard python main.py paper >> "$LOG" 2>&1

echo "=== $(date -u) — Done ===" >> "$LOG"

# Keep only the latest run — delete all older output folders
cd "$APP_DIR/outputs"
ls -dt */ 2>/dev/null | tail -n +2 | xargs -r sudo rm -rf
echo "=== $(date -u) — Old runs cleaned ===" >> "$LOG"

# Prune this script's own logs older than 30 days (~4 weekly runs) -- these had been
# accumulating unbounded since the project's first deploy (106 files, 375MB, back to
# May) because nothing ever cleaned them. The host-level weekly-disk-cleanup.sh cron
# (Docker build cache + journal logs) doesn't know about this path, so it belongs here,
# next to the thing that creates them.
find "$APP_DIR/logs" -name 'paper_*.log' -mtime +30 -delete
echo "=== $(date -u) — Logs older than 30 days pruned ===" >> "$LOG"
