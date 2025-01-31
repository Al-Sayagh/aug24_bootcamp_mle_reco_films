from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from pydantic import Field

class Settings(BaseSettings):
    # Définir la racine du projet en utilisant Path
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    
    # Définir MLFLOW_TRACKING_URI avec une valeur par défaut, pouvant être surchargée par une variable d'environnement
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8081")

    # Construire des chemins relatifs basés sur PROJECT_ROOT
    RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "df_demonstration.csv"

    CLEAN_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed" / "df_clean.csv"
    TRAINSET_PATH: Path = PROJECT_ROOT / "data" / "processed" / "trainset_surprise.pkl"
    USERS_INFO_PATH: Path = PROJECT_ROOT / "data" / "processed" / "users.json"
    OPTIMIZED_PARAMETERS_PATH: Path = PROJECT_ROOT / "data" / "processed" / "svd_optimization.json"
    FILMS_INFO_PATH: Path = PROJECT_ROOT / "data" / "processed" / "films.json"

    MODEL_PATH: Path = PROJECT_ROOT / "models" / "svd_model.joblib"
    
    FILM_EXTRACTOR_LOG_PATH: Path = PROJECT_ROOT / "logs" / "film_extractor.log"
    USER_EXTRACTOR_LOG_PATH: Path = PROJECT_ROOT / "logs" / "user_extractor.log"
    GRIDSEARCH_LOG_PATH: Path = PROJECT_ROOT / "logs" / "gridsearch.log"
    API_LOG_PATH: Path = PROJECT_ROOT / "logs" / "api.log"
    RECOMMENDER_LOG_PATH: Path = PROJECT_ROOT / "logs" / "recommender.log"

    MODEL_METRICS_PATH: Path = PROJECT_ROOT / "metrics" / "model_metrics.json"


    ALLOWED_ORIGINS: list[str] = ["http://localhost:8080", "http://localhost:3000"]

    # Définir les paramètres airflow
    AIRFLOW_WWW_USER_USERNAME: str = Field(
        default="airflow",
        env="_AIRFLOW_WWW_USER_USERNAME"
    )
    AIRFLOW_WWW_USER_PASSWORD: str = Field(
        default="airflow",
        env="_AIRFLOW_WWW_USER_PASSWORD"
    )

    # Postgres
    POSTGRES_USER: str = Field(default="airflow", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="airflow", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="airflow", env="POSTGRES_DB")

    # Configuration de pydantic-settings
    model_config = SettingsConfigDict(
        env_file=".env",            # Charge les variables depuis le fichier .env
        env_file_encoding="utf-8",  # Encodage du fichier
    )

settings = Settings()
