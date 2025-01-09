from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    DATA_PATH: str = "data/raw/df_demonstration.csv"
    MODEL_PATH: str = "models/svd_model.joblib"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()