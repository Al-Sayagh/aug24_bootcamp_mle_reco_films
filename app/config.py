from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Définir la racine du projet en utilisant Path
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    
    # Construire des chemins relatifs basés sur PROJECT_ROOT
    DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "df_demonstration.csv"
    MODEL_PATH: Path = PROJECT_ROOT / "models" / "svd_model.joblib"
    
    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000"]

    # Définir les paramètres airflow
    airflow_uid: str = "501" 
    airflow_gid: str = "0" 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
