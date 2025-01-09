import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from pathlib import Path
import joblib
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import threading
from threading import Lock

from app.config import settings
from app.models import MovieRecommendation, ModelMetrics

# Configuration du logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('recommender.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


class MovieRecommender:
    """Système de recommandation de films utilisant l'algorithme SVD avec gestion des mises à jour."""

    def __init__(self):
        """Initialise le système de recommandation avec gestion du state."""
        self.model = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )
        self.data = None
        self.trainset = None
        self.df = None
        self.metrics = None
        self.last_training_time = None
        self.user_mapping = {}
        self.reader = None
        self.state_lock = Lock()  # Pour la thread-safety
        self._last_update = None
        self._is_training = False

    def needs_update(self) -> bool:
        """Vérifie si le modèle nécessite une mise à jour."""
        if not self._last_update:
            return True
        
        try:
            csv_mtime = Path(settings.DATA_PATH).stat().st_mtime
            return csv_mtime > self._last_update.timestamp()
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de mise à jour: {e}")
            return True

    async def load_data(self) -> None:
        """Charge et prépare les données pour l'entraînement du modèle de manière thread-safe."""
        with self.state_lock:
            try:
                logger.info(f"Début du chargement des données depuis {settings.DATA_PATH}")
                self.df = pd.read_csv(settings.DATA_PATH, dtype={'id_utilisateur': str})

                required_columns = [
                    'id_utilisateur', 'titre_film', 'score_pertinence',
                    'realisateurs_principaux', 'noms_acteurs', 'note_moyenne_imdb'
                ]
                self._verify_columns(required_columns)
                await self._prepare_data()
                self._create_user_mapping()
                self._setup_surprise_reader()
                self._log_data_statistics()
                
                self._last_update = datetime.now()

            except Exception as e:
                logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
                raise

    async def train(self, force: bool = False) -> None:
        """Entraîne le modèle SVD avec gestion de la concurrence."""
        if self._is_training and not force:
            logger.warning("Entraînement déjà en cours, ignoré.")
            return

        with self.state_lock:
            try:
                if not force and not self.needs_update():
                    logger.info("Modèle à jour, pas besoin de réentraînement.")
                    return

                self._is_training = True
                if self.trainset is None:
                    raise ValueError("Les données doivent être chargées avant l'entraînement")

                start_time = datetime.now()
                logger.info("Début de l'entraînement du modèle SVD...")

                self.model.fit(self.trainset)

                # Sauvegarde du modèle
                model_dir = Path(settings.MODEL_PATH).parent
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.model, settings.MODEL_PATH)

                self.last_training_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Modèle entraîné et sauvegardé en {self.last_training_time:.2f} secondes")

            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
                raise
            finally:
                self._is_training = False

    async def _prepare_data(self) -> None:
        """Prépare les données de manière asynchrone."""
        initial_size = len(self.df)

        self.df['id_utilisateur'] = self.df['id_utilisateur'].astype(str)
        self.df['id_film'] = self.df['id_film'].astype(int)
        self.df['score_pertinence'] = pd.to_numeric(self.df['score_pertinence'], errors='coerce')

        self.df = self.df.dropna(subset=['score_pertinence'])
        dropped_rows = initial_size - len(self.df)

        if dropped_rows > 0:
            logger.warning(f"{dropped_rows} lignes supprimées car score_pertinence manquant")

    def _verify_columns(self, required_columns: List[str]) -> None:
        """Vérifie la présence des colonnes requises."""
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le dataset: {missing_columns}")

    def _create_user_mapping(self) -> None:
        """Crée un dictionnaire de mapping pour les IDs utilisateurs."""
        self.user_mapping = {str(uid): True for uid in self.df['id_utilisateur'].unique()}
        logger.info(f"Mapping créé pour {len(self.user_mapping)} utilisateurs uniques")

    def _setup_surprise_reader(self) -> None:
        """Configure le Reader Surprise et prépare les données."""
        score_min = self.df['score_pertinence'].min()
        score_max = self.df['score_pertinence'].max()
        self.reader = Reader(rating_scale=(score_min, score_max))

        self.data = Dataset.load_from_df(
            self.df[['id_utilisateur', 'titre_film', 'score_pertinence']],
            self.reader
        )
        self.trainset = self.data.build_full_trainset()

    def _log_data_statistics(self) -> None:
        """Enregistre les statistiques des données."""
        logger.info(f"Données chargées avec succès: {len(self.df)} évaluations")
        logger.info(f"Nombre d'utilisateurs uniques: {self.df['id_utilisateur'].nunique()}")
        logger.info(f"Nombre de films uniques: {self.df['titre_film'].nunique()}")
        logger.info(f"Score de pertinence moyen: {self.df['score_pertinence'].mean():.2f}")

    async def recommend(self, id_utilisateur: int, n: int = 10) -> List["MovieRecommendation"]:
        """Génère des recommandations personnalisées de manière thread-safe."""
        with self.state_lock:
            try:
                if self.needs_update():
                    logger.info("Mise à jour des données avant recommandation...")
                    await self.load_data()
                    await self.train()

                user_id_str = str(id_utilisateur)
                logger.info(f"Génération de recommandations pour l'utilisateur {user_id_str}")

                if user_id_str not in self.user_mapping:
                    available_users = list(self.user_mapping.keys())[:5]
                    raise ValueError(
                        f"Utilisateur {id_utilisateur} non trouvé. "
                        f"Exemples d'utilisateurs disponibles: {available_users}"
                    )

                films_vus = set(self.df[self.df['id_utilisateur'] == user_id_str]['titre_film'])
                tous_les_films = set(self.df['titre_film'].unique())
                films_non_vus = tous_les_films - films_vus

                recommandations = []
                for film in films_non_vus:
                    prediction = self.model.predict(user_id_str, film)
                    film_data = self.df[self.df['titre_film'] == film].iloc[0]

                    recommandation = MovieRecommendation(
                        titre=film,
                        score_pertinence_predit=prediction.est,
                        realisateur=film_data.get('realisateurs_principaux', None),
                        acteurs=film_data.get('noms_acteurs', None),
                        note_imdb=film_data.get('note_moyenne_imdb', None)
                    )
                    recommandations.append(recommandation)

                recommandations = sorted(
                    recommandations,
                    key=lambda x: (x.score_pertinence_predit, x.note_imdb or 0),
                    reverse=True,
                )[:n]

                logger.info(f"{len(recommandations)} recommandations générées pour l'utilisateur {id_utilisateur}")
                return recommandations

            except Exception as e:
                logger.error(f"Erreur dans recommend(): {e}", exc_info=True)
                raise

    async def evaluate(self) -> ModelMetrics:
        """Évalue les performances du modèle de manière thread-safe."""
        with self.state_lock:
            try:
                logger.info("Évaluation du modèle en cours...")
                results = cross_validate(
                    self.model,
                    self.data,
                    measures=["RMSE", "MAE"],
                    cv=5,
                    verbose=False
                )

                rmse = np.mean(results["test_rmse"])
                mae = np.mean(results["test_mae"])

                metrics = ModelMetrics(
                    rmse=rmse,
                    mae=mae,
                    training_time=self.last_training_time or 0,
                    nombre_utilisateurs=self.trainset.n_users,
                    nombre_films=self.trainset.n_items,
                    score_pertinence_moyen=self.trainset.global_mean
                )

                logger.info(f"Évaluation terminée - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                return metrics

            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation: {e}", exc_info=True)
                raise