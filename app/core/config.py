"""
Application configuration settings
"""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # Project Information
    PROJECT_NAME: str = "FastAPI Application"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "A scalable FastAPI application"
    API_V1_STR: str = "/api/v1"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8005
    DEBUG: bool = False

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8005"]

    # Database
    DATABASE_URL: str = "sqlite:///./app.db"
    DB_ECHO: bool = False

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Redis (optional)
    REDIS_URL: Optional[str] = None

    # LLM / AI
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    VECTOR_STORE_PATH: str = "faiss_index_openai"
    EMBEDDING_API_URL: str = "https://lamhieu-lightweight-embeddings.hf.space/v1/embeddings"
    EMBEDDING_MODEL: str = "bge-m3"
    
    # Pinecone Vector Database
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "cafs-chatbot-memory"
    PINECONE_CLOUD: str = "aws"  # or "gcp", "azure"
    PINECONE_REGION: str = "us-east-1"  # e.g., "us-east-1", "eu-west-1"
    
    # Speech-to-Text (uses OpenAI gpt-4o-transcribe - no config needed)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

