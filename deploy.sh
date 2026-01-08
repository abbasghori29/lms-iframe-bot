#!/bin/bash
# Deployment script for LMS Bot
# This script is run on the EC2 instance

set -e  # Exit on error

APP_DIR="${APP_DIR:-/home/ec2-user/lms-bot}"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="lms-bot"

echo "🚀 Starting deployment..."

# Navigate to app directory
cd "$APP_DIR"

# Pull latest code
echo "📥 Pulling latest code from main branch..."
git fetch origin
git reset --hard origin/main

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Reload systemd and restart service
echo "🔄 Restarting service..."
sudo systemctl daemon-reload
sudo systemctl restart "$SERVICE_NAME" || echo "⚠️  Service might not exist yet"

# Show service status
echo "📊 Service status:"
sudo systemctl status "$SERVICE_NAME" --no-pager || echo "⚠️  Service status unavailable"

echo "✅ Deployment completed successfully!"

