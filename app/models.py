from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class MovieRecommendation(BaseModel):
    titre: str
    score_pertinence_predit: float
    realisateur: Optional[str]
    acteurs: Optional[str]
    note_imdb: Optional[float]

class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    training_time: float
    nombre_utilisateurs: int
    nombre_films: int
    score_pertinence_moyen: float

class User(BaseModel):
    id: str
    nom: str  # Si disponible dans vos données