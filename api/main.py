from pathlib import Path
import sys
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from threading import Lock
import asyncio
import time
import threading
from cachetools import TTLCache, cached
import subprocess
import json
from datetime import datetime
import uvicorn

# Ajouter le répertoire racine au sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import local
from src.config import settings
from src.models import ModelMetrics
from src.recommender import MovieRecommender
from src.gridsearch import main as gridsearch_main

# Configurer le logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_path = settings.API_LOG_PATH
# Crée le répertoire si nécessaire
os.makedirs(os.path.dirname(file_path), exist_ok=True)

file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
))
logger.addHandler(file_handler)

# Ajouter un handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Phase de démarrage
        logger.info("Démarrage de l'application - Initialisation du système")

        logger.info("Phase 1: Chargement des données")
        recommender.load_csv()
        logger.info("Données chargées avec succès")

        logger.info("Phase 1.1: Préparation des données")
        await recommender.prepare_data()
        logger.info("Données préparées avec succès")

        logger.info("Phase 2: Initialisation du gestionnaire d'utilisateurs")
        await user_manager.refresh_users()

        logger.info("Phase 3: Entraînement du modèle")
        await recommender.train(force=True)

        logger.info("Démarrage de la surveillance des fichiers")
        user_manager.watcher.start()

        logger.info("Système initialisé et prêt à l'emploi")
        yield  # Application en cours d'exécution
    except Exception as e:
        logger.critical(f"Erreur critique lors de l'initialisation: {str(e)}", exc_info=True)
        raise RuntimeError(f"Échec de l'initialisation du système: {str(e)}")
    finally:
        # Phase d'arrêt
        cleanup_logging()
        print("Application arrêtée proprement.")
        user_manager.watcher.stop()

# Créer l'application FastAPI avec gestionnaire lifespan
app = FastAPI(
    title="Système de Recommandation de Films",
    description="API de recommandation de films basée sur l'algorithme Surprise SVD.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def cleanup_logging():
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

class UserResponse(BaseModel):
    id: str
    nom: str
    nb_films_vus: Optional[int] = None

class RecommendationRequest(BaseModel):
    n_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Nombre de recommandations souhaitées"
    )

class PaginationParams(BaseModel):
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=100)

class CSVWatcher:
    def __init__(self, csv_path: str, callback, check_interval: int = 300):
        """Initialise le surveillant de fichier CSV."""
        self.csv_path = csv_path
        self.callback = callback
        self.check_interval = check_interval
        self.last_size = 0
        self.last_mtime = 0
        self.should_run = True
        self.lock = Lock()
        self._thread = None
        logger.info(f"CSVWatcher initialisé pour {csv_path}")

    def _get_file_stats(self):
        """Obtient les statistiques actuelles du fichier."""
        try:
            stats = os.stat(self.csv_path)
            return stats.st_size, stats.st_mtime
        except Exception as e:
            logger.error(f"Erreur lors de la lecture des stats du fichier: {e}")
            return 0, 0

    def _check_file(self):
        """Vérifie si le fichier a été modifié."""
        current_size, current_mtime = self._get_file_stats()
        if current_size != self.last_size or current_mtime != self.last_mtime:
            logger.info(f"Modification détectée dans {self.csv_path}")
            with self.lock:
                self.last_size = current_size
                self.last_mtime = current_mtime
                asyncio.run(self.callback())

    def _watch_loop(self):
        """Boucle principale de surveillance."""
        logger.info("Démarrage de la surveillance du fichier CSV")
        self.last_size, self.last_mtime = self._get_file_stats()
        while self.should_run:
            self._check_file()
            time.sleep(self.check_interval)

    def start(self):
        """Démarre la surveillance dans un thread séparé."""
        if self._thread is None or not self._thread.is_alive():
            self.should_run = True
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()
            logger.info("Surveillance du CSV démarrée")

    def stop(self):
        """Arrête la surveillance."""
        try:
            self.should_run = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1)
        except Exception as e:
            print(f"Erreur ignorée lors de l'arrêt de la surveillance: {e}")

class UserManager:
    def __init__(self, recommender: Optional[MovieRecommender] = None):
        """Initialise le gestionnaire d'utilisateurs."""
        self._cache = TTLCache(maxsize=1000, ttl=3600)
        self._users: Dict[str, Dict[str, Any]] = {}
        self._last_update = None
        self.update_lock = Lock()
        self.csv_path = settings.RAW_DATA_PATH
        self.json_path = settings.USERS_INFO_PATH
        self.recommender = recommender

        self.watcher = CSVWatcher(
            csv_path=self.csv_path,
            callback=self._handle_csv_changes,
            check_interval=300
        )

    async def _handle_csv_changes(self):
        """Gère les changements détectés dans le CSV de manière asynchrone."""
        logger.info("Changements détectés dans le CSV - Mise à jour des données")
        try:
            python_path = os.path.join(os.path.dirname(sys.executable), "python")
            script_path = os.path.abspath("src/extract_user_info.py")

            await asyncio.to_thread(lambda: subprocess.run([python_path, script_path], check=True))
            await self._load_from_json()

            if self.recommender:
                logger.info("Mise à jour du système de recommendation...")
                await self.recommender.load_data()
                await self.recommender.train(force=True)

            logger.info("Mise à jour réussie suite aux changements du CSV")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour après changement CSV: {e}")

    async def _load_from_json(self):
        """Charge les données depuis le fichier JSON de manière asynchrone."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._users = {
                user['id']: {
                    'nom': user['nom'],
                    'nb_films_vus': user['nb_films_notes']
                }
                for user in data.get('utilisateurs', [])
            }
            self._last_update = datetime.now()
            self._cache.clear()
            logger.info(f"{len(self._users)} utilisateurs chargés depuis le JSON")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du JSON: {e}")
            raise

    async def refresh_users(self, df=None):
        """Rafraîchit la liste des utilisateurs de manière asynchrone."""
        logger.debug("Début de la mise à jour des utilisateurs")

        async with asyncio.Lock():
            try:
                python_path = os.path.join(os.path.dirname(sys.executable), "python")
                script_path = os.path.abspath("src/extract_user_info.py")

                await asyncio.to_thread(lambda: subprocess.run([python_path, script_path], check=True))
                await self._load_from_json()
                self.watcher.start()

                if self.recommender and df is not None:
                    await self.recommender.load_data()

                logger.info("Mise à jour des utilisateurs terminée avec succès")
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour des utilisateurs: {e}")
                raise

    @cached(cache=TTLCache(maxsize=1000, ttl=3600))
    def get_users(self, skip: int = 0, limit: int = 10) -> List[UserResponse]:
        """Retourne une liste paginée des utilisateurs."""
        logger.debug(f"Récupération des utilisateurs avec skip={skip}, limit={limit}")
        users = sorted(self._users.items())
        return [
            UserResponse(
                id=user_id,
                nom=data['nom'],
                nb_films_vus=data['nb_films_vus']
            )
            for user_id, data in users[skip:skip + limit]
        ]

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'un utilisateur spécifique."""
        logger.debug(f"Récupération de l'utilisateur avec l'ID: {user_id}")
        return self._users.get(str(user_id))

    def __del__(self):
        """Nettoyage lors de la destruction de l'instance."""
        try:
            if hasattr(self, 'watcher'):
                self.watcher.stop()
        except Exception as e:
            print(f"Erreur ignorée lors du nettoyage du cache: {e}")

recommender = MovieRecommender()
user_manager = UserManager(recommender=recommender)

# Définition des endpoints
@app.get("/", tags=["Général"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "Bienvenue dans l'API de Recommandation de Films",
        "documentation": "/docs",
        "statut": "opérationnel",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/users/available", response_model=Dict[str, Any], tags=["Utilisateurs"])
async def get_available_users(
        pagination: PaginationParams = Depends()
):
    """Liste paginée des utilisateurs disponibles."""
    try:
        users = user_manager.get_users(pagination.skip, pagination.limit)
        total_users = len(user_manager._users)

        return {
            "total": total_users,
            "page": (pagination.skip // pagination.limit) + 1,
            "utilisateurs": users,
            "has_more": (pagination.skip + pagination.limit) < total_users
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des utilisateurs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des utilisateurs")

@app.get("/recommendations/{id_utilisateur}", response_model=Dict[str, Any], tags=["Recommandations"])
async def get_recommendations(
        id_utilisateur: str,
        params: RecommendationRequest = Depends()
):
    """Génère des recommandations personnalisées pour un utilisateur."""
    try:
        user = user_manager.get_user(id_utilisateur)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "erreur": f"Utilisateur {id_utilisateur} non trouvé",
                    "suggestion": "Utilisez /users/available pour voir la liste des utilisateurs"
                }
            )

        recommendations = await recommender.recommend(
            id_utilisateur=id_utilisateur,
            n=params.n_recommendations
        )

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "utilisateur": {
                "id": id_utilisateur,
                "nom": user['nom'],
                "films_vus": user['nb_films_vus']
            },
            "recommendations": recommendations
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération des recommandations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=ModelMetrics, tags=["Modèle"])
async def get_model_metrics():
    """Récupère les métriques de performance du modèle."""
    try:
        metrics = await recommender.evaluate()
        return metrics
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gridsearch", tags=["Optimisation"])
def get_optimized_parameters():
    """Obtenir les hyperparamètres optimisés par une GridSearch."""
    try:
        optimized_parameters = gridsearch_main()
        return optimized_parameters
    except Exception as e:
        logger.error(f"Erreur lors de l'optimisation des hyperparamètres: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh", tags=["Système"])
async def refresh_system():
    """Force le rafraîchissement des données et du modèle."""
    try:
        logger.info("Début du rafraîchissement du système.")
        
        # Charger les données CSV
        logger.info("Chargement des données CSV...")
        recommender.load_csv()
        logger.info("Données CSV chargées avec succès.")
        
        # Préparer les données pour l'entraînement
        logger.info("Préparation des données...")
        await recommender.prepare_data()
        logger.info("Données préparées avec succès.")
        
        # Rafraîchir les utilisateurs
        logger.info("Rafraîchissement des utilisateurs...")
        await user_manager.refresh_users()
        logger.info("Utilisateurs rafraîchis avec succès.")
        
        # Entraîner le modèle
        logger.info("Entraînement du modèle...")
        await recommender.train(force=True)
        logger.info("Modèle entraîné avec succès.")
        
        return {
            "status": "success",
            "message": "Système rafraîchi avec succès",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur lors du rafraîchissement: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)