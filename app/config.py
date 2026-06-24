import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configurations
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "https://mock-azure.search.windows.net")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "mock-pinecone-key")

    # Engine Limits
    MAX_GRAPH_STEPS: int = 15
    RATE_LIMIT_TOKENS: int = 2000

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()