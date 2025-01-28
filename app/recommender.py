import logging
import json
import joblib 
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from threading import Lock
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Import local
from app.config import settings
from app.models import MovieRecommendation, ModelMetrics

# Configurer le logging
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

    # Définir les paramètres par défaut optimisés par GridSearch
    DEFAULT_PARAMS = {
        'n_factors': 175,
        'n_epochs': 40,
        'lr_all': 0.003,  # Paramètre optimisé pour RMSE
        'reg_all': 0.07,
        'random_state': 42
    }

    def __init__(self):
        """Initialise le système de recommandation avec les (éventuels) meilleurs paramètres."""
        # Essayer de charger les paramètres optimisés, sinon utiliser les valeurs par défaut
        params = self._load_optimized_parameters()
        
        self.model = SVD(
            n_factors=params.get('n_factors', self.DEFAULT_PARAMS['n_factors']),
            n_epochs=params.get('n_epochs', self.DEFAULT_PARAMS['n_epochs']),
            lr_all=params.get('lr_all', self.DEFAULT_PARAMS['lr_all']),
            reg_all=params.get('reg_all', self.DEFAULT_PARAMS['reg_all']),
            random_state=self.DEFAULT_PARAMS['random_state']
        )
        logger.info(f"Modèle initialisé avec les paramètres: {params if params else self.DEFAULT_PARAMS}")
        
        self.df = None
        self.data = None
        self.trainset = None
        self.metrics = None
        self.last_training_time = None
        self.user_mapping = {}
        self.reader = None
        self.state_lock = Lock()
        self._last_update = None
        self._is_training = False

    def _load_optimized_parameters(self) -> Dict:
        """Charge les paramètres optimisés depuis le fichier JSON."""
        try:
            json_path = Path('data/processed/svd_optimization.json')
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Utiliser les paramètres optimisés pour RMSE par défaut
                    params = data['best_parameters']['rmse']
                    logger.info(f"Paramètres optimisés chargés depuis {json_path}")
                    return params
            else:
                logger.info("Utilisation des paramètres par défaut (fichier d'optimisation non trouvé)")
                return {}
        except Exception as e:
            logger.warning(f"Utilisation des paramètres par défaut (erreur lors du chargement: {e})")
            return {}

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

    def load_csv(self) -> None:
        """
        Lit le CSV depuis `settings.DATA_PATH` et le stocke dans self.df.
        Ne contient que la lecture du CSV, sans la préparation Surprise.
        """
        with self.state_lock:
            try:
                logger.info(f"Début du chargement des données depuis {settings.DATA_PATH}...")
                self.df = pd.read_csv(settings.DATA_PATH, dtype={'id_utilisateur': str})
                self._last_update = datetime.now()
                logger.info(f"CSV chargé: {len(self.df)} lignes.")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
                raise

    async def prepare_data(self) -> None:
        """Prépare les données pour l'entraînement du modèle (contrôle de colonnes, nettoyage, création du trainset) 
        puis sauvegarde le trainset dans data/processed de manière asynchrone et thread-safe."""
        with self.state_lock:
            try:
                if self.df is None:
                    raise ValueError("Le DataFrame est vide (self.df est None). Appelez d'abord load_csv().")

                # Vérifier la présence des colonnes requises
                required_columns = [
                    'id_utilisateur', 'titre_film', 'score_pertinence',
                    'realisateurs_principaux', 'noms_acteurs', 'note_moyenne_imdb'
                ]
                missing_cols = [c for c in required_columns if c not in self.df.columns]
                if missing_cols:
                    raise ValueError(f"Colonnes manquantes dans le dataset: {missing_cols}")

                # Nettoyer / préparer
                initial_size = len(self.df)
                self.df['id_film'] = self.df['id_film'].astype(int)
                self.df['score_pertinence'] = pd.to_numeric(self.df['score_pertinence'], errors='coerce')
                self.df.dropna(subset=['score_pertinence'], inplace=True)
                dropped_rows = initial_size - len(self.df)
                if dropped_rows > 0:
                    logger.warning(f"{dropped_rows} lignes supprimées car score_pertinence manquant")

                # Créer un dictionnaire de mapping pour les IDs utilisateurs.
                self.user_mapping = {str(uid): True for uid in self.df['id_utilisateur'].unique()}
                logger.info(f"Mapping créé pour {len(self.user_mapping)} utilisateurs uniques")

                # Configurer Surprise et préparer la donnée sous forme de trainset
                score_min = self.df['score_pertinence'].min()
                score_max = self.df['score_pertinence'].max()
                self.reader = Reader(rating_scale=(score_min, score_max))

                self.data = Dataset.load_from_df(
                    self.df[['id_utilisateur', 'titre_film', 'score_pertinence']], 
                    self.reader
                )
                self.trainset = self.data.build_full_trainset()

                # Sauvegarder le trainset dans data/processed
                out_path = Path("data/processed/trainset_surprise.pkl")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.trainset, out_path)
                logger.info(f"Trainset sauvegardé dans {out_path}.")

                # Logger les statistiques
                logger.info(f"Données préparées avec succès: {len(self.df)} évaluations.")
                logger.info(f"Nombre d'utilisateurs uniques: {self.df['id_utilisateur'].nunique()}")
                logger.info(f"Nombre de films uniques: {self.df['titre_film'].nunique()}")
                logger.info(f"Score de pertinence moyen: {self.df['score_pertinence'].mean():.2f}")

            except Exception as e:
                logger.error(f"Erreur lors de la préparation des données: {e}", exc_info=True)
                raise

    async def train(self, force: bool = False) -> None:
        """Entraîne le modèle SVD et l'enregistre, avec gestion de la concurrence."""
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
                    raise ValueError("Les données doivent être préparées avant l'entraînement (self.trainset est None).")

                start_time = datetime.now()
                logger.info("Début de l'entraînement du modèle SVD...")

                self.model.fit(self.trainset)

                # Enregistrer dans Mlflow
                mlflow.set_experiment("Movie Recommandation System")
                with mlflow.start_run(run_name="Train-SVD-{0}".format(start_time.strftime('%Y%m%d-%H%M%S'))):
                    # Logger les paramètres du modèle
                    mlflow.log_params(self.model.__dict__)
                    # Logger la durée d'entraînement
                    training_duration = (datetime.now() - start_time).total_seconds()
                    mlflow.log_metric("training_time", training_duration)
                    # Sauvegarder le modèle
                    model_dir = Path(settings.MODEL_PATH).parent
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(self.model, settings.MODEL_PATH)
                    # Logger le modèle dans MLflow
                    mlflow.sklearn.log_model(self.model, artifact_path="SVD-model")
                    
                self.last_training_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Modèle entraîné et sauvegardé en {self.last_training_time:.2f} secondes")

            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
                raise
            finally:
                self._is_training = False

    async def recommend(self, id_utilisateur: int, n: int = 10) -> List["MovieRecommendation"]:
        """Génère des recommandations personnalisées de manière thread-safe (exige trainset + modèle valides)."""
        with self.state_lock:
            try:
                if self.needs_update():
                    logger.info("Mise à jour des données avant recommandation...")
                    self.load_csv()
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

                    # Exemple d'objet MovieRecommendation
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
        """Évalue les performances du modèle de manière thread-safe via cross-validation,
        puis sauvegarde les metrics dans un fichier JSON sous /metrics/."""
        with self.state_lock:
            try:
                logger.info("Évaluation du modèle en cours...")
                if self.data is None or self.trainset is None:
                    raise ValueError("Les données doivent être préparées avant l'évaluation.")
                
                results = cross_validate(
                    self.model,
                    self.data,
                    measures=["RMSE", "MAE"],
                    cv=5,
                    verbose=False
                )

                rmse = float(np.mean(results["test_rmse"]))
                mae = float(np.mean(results["test_mae"]))

                metrics = ModelMetrics(
                    rmse=rmse,
                    mae=mae,
                    training_time=self.last_training_time or 0,
                    nombre_utilisateurs=self.trainset.n_users,
                    nombre_films=self.trainset.n_items,
                    score_pertinence_moyen=self.trainset.global_mean
                )

                # Sauvegarde des métriques au format JSON
                metrics_dir = Path("metrics")
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / "model_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics.dict(), f, indent=4, ensure_ascii=False)

                logger.info(f"Métriques sauvegardées dans {metrics_path}")   
                logger.info(f"Évaluation terminée - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
                return metrics

            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation: {e}", exc_info=True)
                raise