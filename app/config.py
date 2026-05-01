"""
Configuration loader for the RAG project.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env or .env.example
load_dotenv(dotenv_path=".env", override=True)

class Config:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "langgraph_states")
    POSTGRES_SCHEMA: str = os.getenv("POSTGRES_SCHEMA", "public")
    
    # Database URL construction
    @property
    def DATABASE_URL(self) -> str:
        """Get the database connection URL."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            db_url = (
                f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        return db_url
    
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """Get the async database connection URL."""
        # First check for a specific async URL
        db_url = os.getenv("ASYNC_DATABASE_URL")
        if db_url:
            return db_url
            
        # If not, check for DATABASE_URL but ensure it has the right driver
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            # If DATABASE_URL is provided but doesn't have asyncpg, replace the driver
            if "postgresql+asyncpg://" not in db_url:
                # Replace postgresql:// or postgresql+psycopg2:// with postgresql+asyncpg://
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
                db_url = db_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
            return db_url
            
        # If no environment variables are set, build the URL manually
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def USE_OLLAMA(self) -> bool:
        """Check if Ollama is configured (OLLAMA_MODEL is set)."""
        return bool(self.OLLAMA_MODEL)
    
    @property
    def model_name(self) -> str:
        """Get the name of the model being used."""
        model_name = self.OLLAMA_MODEL if self.USE_OLLAMA else self.OPENROUTER_MODEL
        return model_name

config = Config()