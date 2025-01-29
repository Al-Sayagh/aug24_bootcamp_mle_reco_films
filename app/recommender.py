import logging
from surprise import SVD, Dataset, Reader
from threading import Lock
from typing import List, Dict
import json
import pandas as pd
from datetime import datetime
import joblib 
from mlflow.tracking import MlflowClient
import mlflow
from surprise.model_selection import cross_validate
import numpy as np

# Import local
from app.config import settings
from app.models import MovieRecommendation, ModelMetrics

# Configurer le logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_path = settings.RECOMMENDER_LOG_PATH
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
            json_path = settings.OPTIMIZED_PARAMETERS_PATH
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
            csv_mtime = settings.RAW_DATA_PATH.stat().st_mtime
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
                logger.info(f"Début du chargement des données depuis {settings.RAW_DATA_PATH}...")
                self.df = pd.read_csv(settings.RAW_DATA_PATH, dtype={'id_utilisateur': str})
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
                logger.info("Début de la préparation des données.")
                            
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
                logger.info("Trainset créé.")

                # Sauvegarder le trainset dans data/processed
                out_path = settings.TRAINSET_PATH
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
                logger.info("Début de l'entraînement du modèle SVD...")
                
                start_time = datetime.now()
                run_name = f"train-{start_time.strftime('%Y%m%d-%H%M%S')}"
                self.model.fit(self.trainset)
                end_time = datetime.now()
                logger.info("Entraînement du modèle SVD terminé.")

                # Calculer la durée d'entraînement
                training_duration = (end_time - start_time).total_seconds()

                # Configuration de MLflow
                mlflow_tracking_uri = settings.MLFLOW_TRACKING_URI
                mlflow.set_tracking_uri(mlflow_tracking_uri)

                # Initialiser le client MLflow
                client = MlflowClient()

                experiment_name = "train_svd_surprise"
                
                # Vérifier si l'expérience existe et est active
                try:
                    experiment = client.get_experiment_by_name(experiment_name)
                    if experiment and experiment.lifecycle_stage == "deleted":
                        client.restore_experiment(experiment.experiment_id)
                        logger.info(f"L'expérience '{experiment_name}' a été restaurée.")
                    elif not experiment:
                        # Créer une nouvelle expérience si elle n'existe pas
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Nouvelle expérience '{experiment_name}' créée.")
                    else:
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Utilisation de l'expérience existante '{experiment_name}'.")
                except Exception as e:
                    logger.error(f"Erreur lors de la gestion de l'expérience MLflow: {e}", exc_info=True)
                    raise

                with mlflow.start_run(run_name=run_name):
                    # Créer un dictionnaire des paramètres manuellement
                    model_params = {
                        "n_factors": self.model.n_factors,
                        "n_epochs": self.model.n_epochs,
                        "random_state": self.model.random_state
                    }
                    # Logger les paramètres du modèle
                    mlflow.log_params(model_params)

                    # Logger la durée d'entraînement
                    mlflow.log_metric("training_time", training_duration)
                    
                    # Sauvegarder le modèle
                    model_dir = settings.MODEL_PATH.parent
                    model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(self.model, settings.MODEL_PATH)
                    logger.info(f"Modèle sauvegardé dans {settings.MODEL_PATH}.")

                    # Loguer le modèle comme artifact
                    mlflow.log_artifact(str(settings.MODEL_PATH), artifact_path="model")

                    # Loguer le fichier log comme artifact
                    mlflow.log_artifact(str(file_path), artifact_path="log")

                    
                self.last_training_time = training_duration
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
        puis sauvegarde les metrics dans un fichier JSON sous /metrics/ et les log dans MLflow."""
        with self.state_lock:
            try:
                logger.info("Évaluation du modèle en cours...")
                start_time = datetime.now()
                run_name = f"evaluate-{start_time.strftime('%Y%m%d-%H%M%S')}"
                if self.data is None or self.trainset is None:
                    raise ValueError("Les données doivent être préparées avant l'évaluation.")
                
                # Effectuer la cross-validation
                results = cross_validate(
                    self.model,
                    self.data,
                    measures=["RMSE", "MAE"],
                    cv=5,
                    verbose=False
                )

                # Calculer les moyennes des métriques
                rmse = float(np.mean(results["test_rmse"]))
                mae = float(np.mean(results["test_mae"]))

                # Créer l'objet ModelMetrics
                metrics = ModelMetrics(
                    rmse=rmse,
                    mae=mae,
                    training_time=self.last_training_time or 0,
                    nombre_utilisateurs=self.trainset.n_users,
                    nombre_films=self.trainset.n_items,
                    score_pertinence_moyen=self.trainset.global_mean
                )

                # Sauvegarde des métriques au format JSON
                metrics_dir = settings.MODEL_METRICS_PATH.parent
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = settings.MODEL_METRICS_PATH
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics.dict(), f, indent=4, ensure_ascii=False)

                logger.info(f"Métriques sauvegardées dans {metrics_path}")   
                logger.info(f"Évaluation terminée - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
                # Récupérer l'URI de tracking MLflow depuis une variable d'environnement ou une connexion Airflow
                mlflow_tracking_uri = settings.MLFLOW_TRACKING_URI
                mlflow.set_tracking_uri(mlflow_tracking_uri)

                # Initialiser le client MLflow
                client = MlflowClient()

                experiment_name = "evaluate_svd_surprise"

                # Vérifier si l'expérience existe et est active
                try:
                    experiment = client.get_experiment_by_name(experiment_name)
                    if experiment and experiment.lifecycle_stage == "deleted":
                        client.restore_experiment(experiment.experiment_id)
                        logger.info(f"L'expérience '{experiment_name}' a été restaurée.")
                    elif not experiment:
                        # Créer une nouvelle expérience si elle n'existe pas
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Nouvelle expérience '{experiment_name}' créée.")
                    else:
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Utilisation de l'expérience existante '{experiment_name}'.")
                except Exception as e:
                    logger.error(f"Erreur lors de la gestion de l'expérience MLflow: {e}", exc_info=True)
                    raise

                with mlflow.start_run(run_name=run_name):
                    # Loguer les métriques dans MLflow
                    mlflow.log_metrics({
                        "RMSE": rmse,
                        "MAE": mae
                    })
                
                    # Loguer les paramètres pertinents
                    mlflow.log_params({
                        "training_time": self.last_training_time or 0,
                        "nombre_utilisateurs": self.trainset.n_users,
                        "nombre_films": self.trainset.n_items,
                        "score_pertinence_moyen": self.trainset.global_mean
                    })
                
                    # Loguer le fichier JSON des métriques comme artifact
                    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

                    # Loguer le fichier log comme artifact
                    mlflow.log_artifact(str(file_path), artifact_path="log")

                return metrics

            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation: {e}", exc_info=True)
                raise