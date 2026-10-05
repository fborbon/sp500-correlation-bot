#!/bin/bash
# Initial EC2 setup script — run once as ubuntu user on a fresh Ubuntu 24.04 instance.
# Usage: bash deploy.sh

set -e

REPO="https://github.com/fborbon/SP500_Invest.git"
APP_DIR="/opt/forwardforecasting"
DOMAIN="forwardforecasting.eu"

echo "=== [1/7] System update ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== [2/7] Install Docker ==="
sudo apt-get install -y ca-certificates curl gnupg nginx certbot python3-certbot-nginx git
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker ubuntu
newgrp docker

echo "=== [3/7] Clone repo ==="
sudo mkdir -p "$APP_DIR"
sudo chown ubuntu:ubuntu "$APP_DIR"
git clone "$REPO" "$APP_DIR"
mkdir -p "$APP_DIR/cache" "$APP_DIR/outputs" "$APP_DIR/logs"

echo "=== [4/7] Build Docker image ==="
cd "$APP_DIR"
docker compose build

echo "=== [5/7] Start dashboard ==="
docker compose up -d

echo "=== [6/7] Configure Nginx ==="
sudo cp "$APP_DIR/nginx/forwardforecasting.conf" /etc/nginx/sites-available/forwardforecasting.conf
sudo ln -sf /etc/nginx/sites-available/forwardforecasting.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

echo "=== [7/7] SSL certificate (Let's Encrypt) ==="
# Make sure DNS A record is already pointing to this server before running this step.
sudo certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" --non-interactive --agree-tos -m forwardforecasting@gmail.com

echo ""
echo "=== Setup complete ==="
echo "Dashboard running at https://www.$DOMAIN"
echo ""
echo "Next: set up the weekly cron job (matches the validated 7-trading-day rebalance"
echo "cadence -- NOT daily, see scripts/run_paper_daily.sh for why):"
echo "  chmod +x $APP_DIR/scripts/run_paper_daily.sh"
echo "  crontab -e"
echo "  Add:  15 16 * * 1  $APP_DIR/scripts/run_paper_daily.sh"
