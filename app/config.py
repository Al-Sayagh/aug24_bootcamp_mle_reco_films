from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Définir le répertoire racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Chemins dynamiques basés sur le répertoire racine
    DATA_PATH: str = str(BASE_DIR / "data/raw/df_demonstration.csv")
    MODEL_PATH: str = str(BASE_DIR / "models/svd_model.joblib")
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000"]

    # Chargement des configurations depuis un fichier .env
    model_config = SettingsConfigDict(env_file=".env")

# Instanciation des paramètres
settings = Settings()
