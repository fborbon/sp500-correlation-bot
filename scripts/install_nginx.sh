#!/usr/bin/env bash
# Installs this repo's nginx files for the main site (forwardforecasting.eu) on the EC2 host.
# This repo is the source of truth for that server block, including the central-login portal
# (Authelia) and every protected location. Called by .github/workflows/deploy.yml after the
# checkout moved from OLD_REV to the new HEAD.
#
# Drift guard: a file is only replaced if the live copy still equals what OLD_REV shipped.
# If someone edited the live file by hand, the deploy stops and prints the diff instead of
# silently overwriting it (commit the live change to this repo, then redeploy).
set -euo pipefail
OLD_REV="${1:?usage: install_nginx.sh <previously deployed git rev>}"
cd "$(dirname "$0")/.."

FILES=(
  "nginx/forwardforecasting.conf:/etc/nginx/sites-available/forwardforecasting.conf"
  "nginx/snippets/globalnews-nocache.conf:/etc/nginx/snippets/globalnews-nocache.conf"
  "nginx/snippets/traffic-analytics-nocache.conf:/etc/nginx/snippets/traffic-analytics-nocache.conf"
)
BACKUP_DIR=/etc/nginx/backups
TS=$(date +%Y%m%d%H%M%S)
changed=()

for pair in "${FILES[@]}"; do
  src="${pair%%:*}"; dst="${pair#*:}"
  if sudo cmp -s "$src" "$dst"; then continue; fi                 # already live
  old=$(mktemp); git show "$OLD_REV:$src" > "$old" 2>/dev/null || : > "$old"
  if [ -s "$old" ] && ! sudo cmp -s "$old" "$dst"; then
    echo "✗ DRIFT: $dst was edited on the host since the last deploy. Not overwriting."
    sudo diff -u "$old" "$dst" | head -60 || true
    rm -f "$old"; exit 1
  fi
  rm -f "$old"
  sudo mkdir -p "$BACKUP_DIR"
  [ -f "$dst" ] && sudo cp "$dst" "$BACKUP_DIR/$(basename "$dst").bak-$TS"
  sudo install -m 644 "$src" "$dst"
  changed+=("$dst")
done

# The site must be enabled via symlink (one canonical copy in sites-available)
if [ ! -L /etc/nginx/sites-enabled/forwardforecasting.conf ]; then
  echo "✗ /etc/nginx/sites-enabled/forwardforecasting.conf is not a symlink to sites-available"; exit 1
fi

if [ ${#changed[@]} -eq 0 ]; then echo "nginx: no changes"; exit 0; fi
if sudo nginx -t; then
  sudo systemctl reload nginx
  echo "nginx: installed ${changed[*]} and reloaded"
else
  echo "✗ nginx -t failed, restoring previous files"
  for dst in "${changed[@]}"; do
    b="$BACKUP_DIR/$(basename "$dst").bak-$TS"
    if [ -f "$b" ]; then sudo cp "$b" "$dst"; else sudo rm -f "$dst"; fi
  done
  sudo nginx -t && sudo systemctl reload nginx
  exit 1
fi
