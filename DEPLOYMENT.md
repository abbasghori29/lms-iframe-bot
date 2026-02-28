# Deployment Architecture

## Overview

This application uses a **split deployment** architecture:

- **Backend**: Runs locally on EC2 (localhost:8005), not exposed to the internet
- **Frontend**: Deployed via Docker, exposed on the domain via Nginx
- **Nginx**: Serves frontend and proxies `/api/*` requests to backend localhost

## Architecture Diagram

```
Internet
   │
   ▼
[EC2 Instance]
   │
   ├── Nginx (Port 443/80)
   │   ├── Serves Frontend (Docker container on port 3000)
   │   └── Proxies /api/* → Backend (localhost:8005)
   │
   ├── Frontend (Docker)
   │   └── Next.js app on port 3000
   │   └── NEXT_PUBLIC_API_URL=/api (relative path)
   │
   └── Backend (Systemd Service)
       └── FastAPI on 127.0.0.1:8005 (localhost only)
```

## Components

### Backend (FastAPI)
- **Location**: `/home/ec2-user/lms-bot` (or `EC2_APP_DIR`)
- **Service**: `lms-bot.service` (systemd)
- **Port**: `127.0.0.1:8005` (localhost only, not exposed)
- **CORS**: Only allows `localhost:3000` and `127.0.0.1:3000`

### Frontend (Next.js)
- **Location**: `/home/ec2-user/lms-bot/frontend`
- **Deployment**: Docker container (`cafs-frontend`)
- **Port**: `3000` (internal, proxied by Nginx)
- **API URL**: `/api` (relative path, goes through Nginx)

### Nginx
- **Domain**: Configured via `FRONTEND_DOMAIN` secret (default: `api.onlinece.ca`)
- **SSL**: Let's Encrypt via Certbot
- **Routes**:
  - `/api/*` → `http://127.0.0.1:8005/*` (backend proxy)
  - `/` → `http://127.0.0.1:3000` (frontend proxy)

## GitHub Secrets Required

1. `EC2_SSH_KEY` - SSH private key for EC2 access
2. `EC2_HOST` - EC2 instance IP or hostname
3. `EC2_USER` - EC2 username (usually `ec2-user`)
4. `EC2_APP_DIR` - (Optional) Application directory path
5. `FRONTEND_DOMAIN` - (Optional) Frontend domain (default: `api.onlinece.ca`)
6. `GROQ_API_KEY` - Groq API key
7. `OPENAI_API_KEY` - OpenAI API key
8. `PINECONE_API_KEY` - Pinecone API key

## Deployment Process

The GitHub Actions workflow (`deploy.yml`) automatically:

1. **Installs prerequisites**:
   - Python 3.11, Node.js (via Docker), Nginx, Certbot
   - Docker and Docker Compose

2. **Deploys Backend**:
   - Clones/pulls latest code
   - Sets up Python virtual environment
   - Creates `.env` file with API keys
   - Creates systemd service (`lms-bot.service`)
   - Starts backend on `127.0.0.1:8005`

3. **Deploys Frontend**:
   - Builds Docker image with `NEXT_PUBLIC_API_URL=/api`
   - Runs container via `docker-compose`
   - Exposes on port 3000 (internal)

4. **Configures Nginx**:
   - Creates config for frontend domain
   - Sets up SSL via Certbot
   - Proxies `/api/*` to backend
   - Serves frontend on `/`

## Security

- **Backend is NOT exposed** to the internet (localhost only)
- **Frontend** is the only public-facing service
- **API calls** go through Nginx proxy (same domain, no CORS issues)
- **SSL/TLS** via Let's Encrypt (auto-renewing)

## DNS Configuration

Before deployment, ensure your DNS is configured:

- **A Record**: `api.onlinece.ca` → Your EC2 instance's public IP address
- This allows Certbot to verify domain ownership and issue SSL certificates

## EC2 Security Group

Only these ports should be open:

- **Port 443** (HTTPS) - Type: HTTPS, Source: 0.0.0.0/0
- **Port 80** (HTTP) - Type: HTTP, Source: 0.0.0.0/0
- **Port 22** (SSH) - Type: SSH, Source: Your IP (optional, for manual access)

**DO NOT** expose port 8005 to the internet (backend is localhost only).

## Manual Deployment

If you need to deploy manually:

```bash
# SSH into EC2
ssh ec2-user@<EC2_HOST>

# Navigate to app directory
cd /home/ec2-user/lms-bot

# Pull latest code
git pull origin main

# Restart backend
sudo systemctl restart lms-bot

# Rebuild and restart frontend
cd frontend
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Reload Nginx
sudo nginx -t
sudo systemctl reload nginx
```

## Troubleshooting

### Backend not starting
```bash
sudo systemctl status lms-bot
sudo journalctl -u lms-bot -n 50
```

### Frontend not starting
```bash
cd /home/ec2-user/lms-bot/frontend
docker-compose logs
docker ps | grep cafs-frontend
```

### Nginx issues
```bash
sudo nginx -t
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

### Check ports
```bash
sudo netstat -tulpn | grep -E '8005|3000|443|80'
```

## Local Development

For local development, the frontend should point to your local backend:

```bash
# In frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8005
```

The production Docker build uses `/api` (relative path) so it works through Nginx.

