#!/bin/bash
# Initial EC2 setup script
# Run this ONCE on your EC2 instance to set up the environment

set -e

echo "🚀 Setting up EC2 instance for LMS Bot..."

# Update system
echo "📦 Updating system packages..."
sudo yum update -y

# Install Python 3.11 and required packages
echo "🐍 Installing Python 3.11..."
sudo yum install -y python3.11 python3.11-pip python3.11-devel git

# Install system dependencies for Python packages
echo "📦 Installing build dependencies..."
sudo yum install -y gcc gcc-c++ make cmake
sudo yum install -y ffmpeg ffmpeg-devel  # For faster-whisper/av

# Create app directory
APP_DIR="/home/ec2-user/lms-bot"
echo "📁 Creating app directory: $APP_DIR"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Clone repository (if not already cloned)
if [ ! -d ".git" ]; then
    echo "📥 Cloning repository..."
    # You'll need to set up your git remote here
    # git clone <your-repo-url> .
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Copy systemd service file
echo "⚙️  Setting up systemd service..."
sudo cp lms-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lms-bot

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file template..."
    cat > .env << EOF
# Application Settings
PROJECT_NAME=LMS Bot
VERSION=0.1.0
DEBUG=False
HOST=0.0.0.0
PORT=8005

# Database
DATABASE_URL=sqlite:///./app.db

# Security
SECRET_KEY=change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8005"]

# LLM / AI
GROQ_API_KEY=your-groq-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
VECTOR_STORE_PATH=faiss_index_openai
EMBEDDING_API_URL=https://lamhieu-lightweight-embeddings.hf.space/v1/embeddings
EMBEDDING_MODEL=bge-m3

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=cafs-chatbot-memory
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Speech-to-Text
WHISPER_MODEL=base
EOF
    echo "⚠️  Please edit .env file with your actual configuration!"
fi

# Set proper permissions
echo "🔒 Setting permissions..."
chmod +x deploy.sh
chown -R ec2-user:ec2-user "$APP_DIR"

echo "✅ EC2 setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration: nano $APP_DIR/.env"
echo "2. Start the service: sudo systemctl start lms-bot"
echo "3. Check status: sudo systemctl status lms-bot"
echo "4. View logs: sudo journalctl -u lms-bot -f"

