# FastAPI Scalable Application

A production-ready, scalable FastAPI application structure with best practices.

## Features

- ✅ Clean architecture with separation of concerns
- ✅ Database models with SQLAlchemy
- ✅ Pydantic schemas for request/response validation
- ✅ Service layer for business logic
- ✅ Dependency injection
- ✅ Error handling and custom exceptions
- ✅ Logging configuration
- ✅ CORS middleware
- ✅ Environment-based configuration
- ✅ Type hints throughout
- ✅ **RAG-powered Chatbot** with FAISS vector database
- ✅ **LangGraph conversation handling** with smart follow-up support
- ✅ **OpenAI GPT-4o** for LLM and embeddings
- ✅ **Speech-to-Text** using OpenAI gpt-4o-transcribe
- ✅ **PDF document processing** and vectorization
- ✅ **Pinecone** for conversation memory

## Project Structure

```
app/
├── __init__.py
├── main.py                 # Application entry point
├── api/                    # API layer
│   └── v1/
│       ├── router.py       # Main API router
│       └── endpoints/      # API endpoints
│           └── example.py
├── core/                   # Core functionality
│   ├── config.py          # Configuration settings
│   ├── exceptions.py      # Custom exceptions
│   ├── logging.py         # Logging setup
│   └── middleware.py      # Custom middleware
├── db/                     # Database layer
│   ├── database.py        # Database connection
│   └── models/            # SQLAlchemy models
│       ├── base.py
│       └── example.py
├── schemas/                # Pydantic schemas
│   └── example.py
└── services/               # Business logic layer
    └── example.py
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```env
PROJECT_NAME=FastAPI Application
VERSION=0.1.0
DEBUG=True
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO

# LLM / AI Configuration
OPENAI_API_KEY=your-openai-api-key-here
GROQ_API_KEY=your-groq-api-key-here  # Optional fallback
VECTOR_STORE_PATH=faiss_index_openai

# Pinecone (for conversation memory)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=cafs-chatbot-memory
```

## Running the Application

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8005
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8005 --workers 4
```

## Deployment to EC2

This project includes automated deployment to AWS EC2 using GitHub Actions.

### Prerequisites

1. **EC2 Instance**: Running Amazon Linux 2023 (or similar)
2. **GitHub Repository**: Code pushed to GitHub
3. **SSH Access**: EC2 key pair (`lms-bot-key.pem`)

### Initial EC2 Setup (One-time)

1. **SSH into your EC2 instance**:
   ```bash
   ssh -i lms-bot-key.pem ec2-user@16.170.245.197
   ```

2. **Clone the repository**:
   ```bash
   cd /home/ec2-user
   git clone <your-github-repo-url> lms-bot
   cd lms-bot
   ```

3. **Run the setup script**:
   ```bash
   chmod +x setup-ec2.sh
   ./setup-ec2.sh
   ```

4. **Configure environment variables**:
   ```bash
   nano .env
   # Add your API keys and configuration
   ```

5. **Start the service**:
   ```bash
   sudo systemctl start lms-bot
   sudo systemctl enable lms-bot  # Enable auto-start on boot
   ```

6. **Check service status**:
   ```bash
   sudo systemctl status lms-bot
   ```

7. **View logs**:
   ```bash
   sudo journalctl -u lms-bot -f
   ```

### GitHub Actions Setup

1. **Add GitHub Secrets**:
   Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

   Add these **6 secrets**:
   
   **Deployment Secrets:**
   - `EC2_SSH_KEY`: Contents of your private key file (`lms-bot-key.pem`)
   - `EC2_HOST`: Your EC2 public IP or DNS (e.g., `16.170.245.197`)
   - `EC2_USER`: SSH username (usually `ec2-user` for Amazon Linux)
   - `EC2_APP_DIR`: Application directory path (optional, default: `/home/ec2-user/lms-bot`)
   
   **API Keys (automatically added to .env on EC2):**
   - `OPENAI_API_KEY`: Your OpenAI API key (required for LLM, embeddings, and speech-to-text)
   - `GROQ_API_KEY`: Your Groq API key (optional fallback)
   - `PINECONE_API_KEY`: Your Pinecone API key (for conversation memory)

2. **How to get your SSH key content**:
   ```bash
   # On Windows (PowerShell)
   Get-Content lms-bot-key.pem | Out-String
   
   # On Linux/Mac
   cat lms-bot-key.pem
   ```
   Copy the entire output including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`

3. **Note**: The workflow will automatically create/update the `.env` file on EC2 with your API keys from GitHub secrets. No manual setup needed!

### Automated Deployment

Once set up, deployment happens automatically:

- **Trigger**: Every push to `main` branch
- **Manual trigger**: Go to Actions tab → "Deploy to EC2" → "Run workflow"

The workflow will:
1. ✅ Checkout code
2. ✅ SSH into EC2
3. ✅ Pull latest changes
4. ✅ Install/update dependencies
5. ✅ Restart the service
6. ✅ Perform health check

### Managing the Service

**Start service**:
```bash
sudo systemctl start lms-bot
```

**Stop service**:
```bash
sudo systemctl stop lms-bot
```

**Restart service**:
```bash
sudo systemctl restart lms-bot
```

**View logs**:
```bash
sudo journalctl -u lms-bot -f
```

**Check status**:
```bash
sudo systemctl status lms-bot
```

### Security Group Configuration

Ensure your EC2 security group allows:
- **Port 8005** (HTTP): For API access
- **Port 22** (SSH): For deployment (restrict to your IP)

### Troubleshooting

**Service won't start**:
```bash
# Check logs
sudo journalctl -u lms-bot -n 50

# Check if port is in use
sudo netstat -tulpn | grep 8005

# Verify .env file exists
ls -la /home/ec2-user/lms-bot/.env
```

**Deployment fails**:
- Verify GitHub secrets are correct
- Check EC2 security group allows SSH from GitHub Actions IPs
- Verify the app directory path in `EC2_APP_DIR` secret

**Application not accessible**:
- Check security group allows port 8005
- Verify service is running: `sudo systemctl status lms-bot`
- Check application logs: `sudo journalctl -u lms-bot -f`

## API Documentation

Once the server is running, access the interactive API documentation:

- Swagger UI: http://localhost:8005/docs
- ReDoc: http://localhost:8005/redoc

## Chatbot Setup

### 1. Train Vector Database

First, train the vector database from your PDF documents:

```bash
python train_vector_db.py
```

This will:
- Load PDF files (`FIC-CSI-EN-2025.pdf` and `PDF complet de CSI.pdf`)
- Split them into chunks
- Generate embeddings
- Create and save FAISS vector store to `faiss_index/`

### 2. Set Up Environment

Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your-groq-api-key-here
```

### 3. Start the Server

```bash
uvicorn app.main:app --reload
```

### 4. Test the Chatbot

**Using the API:**

```bash
curl -X POST "http://localhost:8005/api/v1/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is CSI?",
    "k": 5
  }'
```

**Using Python:**

```bash
python test_chat_api.py
```

**Using Swagger UI:**

Visit http://localhost:8005/docs and use the `/api/v1/chat/chat` endpoint.

### Chat API Endpoints

- `POST /api/v1/chat/chat` - Chat with the RAG-powered assistant
- `GET /api/v1/chat/health` - Check chatbot service health

### Speech-to-Text API

The speech-to-text service uses OpenAI's `gpt-4o-transcribe` model for accurate transcription.

**Transcribe audio:**
```bash
curl -X POST "http://localhost:8005/api/v1/speech/transcribe" \
  -F "audio=@audio.mp3"
```

**With specific language:**
```bash
curl -X POST "http://localhost:8005/api/v1/speech/transcribe" \
  -F "audio=@audio.mp3" \
  -F "language=fr"
```

**Endpoints:**
- `POST /api/v1/speech/transcribe` - Transcribe audio file to text
- `GET /api/v1/speech/health` - Check speech service health

## Adding New Features

1. **Create a Model** (`app/db/models/your_model.py`):
   - Inherit from `BaseModel`
   - Define your database schema

2. **Create Schemas** (`app/schemas/your_model.py`):
   - Create `Base`, `Create`, `Update`, and `Response` schemas

3. **Create Service** (`app/services/your_model.py`):
   - Implement business logic
   - Handle database operations

4. **Create Endpoints** (`app/api/v1/endpoints/your_model.py`):
   - Define API routes
   - Use dependency injection for database sessions

5. **Register Router** (`app/api/v1/router.py`):
   - Include your new router

## Database Migrations

This project uses Alembic for database migrations. To set up:

```bash
# Initialize Alembic
alembic init alembic

# Create a migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head
```

## Testing

Run tests with pytest:

```bash
pytest
```

## Best Practices

- Keep business logic in services, not in endpoints
- Use Pydantic schemas for all request/response validation
- Use dependency injection for database sessions
- Handle errors with custom exceptions
- Use type hints throughout
- Follow RESTful API conventions

## License

MIT

