import logging
import numpy as np
from datetime import datetime
import pandas as pd
from surprise import SVD, Dataset, Reader
from mlflow.tracking import MlflowClient
import mlflow
from surprise.model_selection import GridSearchCV
import json
from pathlib import Path

# Import local
from app.config import settings

# Configurer le logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_path = settings.GRIDSEARCH_LOG_PATH
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.INFO)
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

class SVDOptimizer:
    def __init__(self):
        # Définir la grille optimisée basée sur les meilleurs résultats précédents
        self.param_grid = {
            'n_factors': [150, 175],    # Centré autour du meilleur résultat (150)
            'n_epochs': [40, 45],       # Centré autour du meilleur résultat (40)
            'lr_all': [0.002, 0.003],   # Centré autour du meilleur résultat (0.002)
            'reg_all': [0.06, 0.07]     # Centré autour du meilleur résultat (0.06)
        }
        self.best_params_rmse = None
        self.best_params_mae = None
        self.best_score_rmse = None
        self.best_score_mae = None
        self.cv_results = None
        
        # Calculer le nombre total de combinaisons
        self.total_combinations = np.prod([len(values) for values in self.param_grid.values()])
        logger.info(f"Nombre total de combinaisons à tester: {self.total_combinations}")
        
        # Logger les paramètres de recherche
        logger.info("Espace de recherche des paramètres (optimisé):")
        for param, values in self.param_grid.items():
            logger.info(f"  {param}: {values}")

    def load_data(self) -> None:
        """Charge les données depuis le CSV."""
        start_time = datetime.now()

        try:
            logger.info("Début du chargement des données...")
            df = pd.read_csv(settings.RAW_DATA_PATH, dtype={'id_utilisateur': str})
            logger.info(f"Dimensions du DataFrame: {df.shape}")
            logger.info(f"Nombre d'utilisateurs uniques: {df['id_utilisateur'].nunique()}")
            logger.info(f"Nombre de films uniques: {df['titre_film'].nunique()}")
            
            # Afficher les statistiques sur les scores de pertinence
            logger.info("Statistiques des scores de pertinence:")
            logger.info(f"  Min: {df['score_pertinence'].min():.2f}")
            logger.info(f"  Max: {df['score_pertinence'].max():.2f}")
            logger.info(f"  Moyenne: {df['score_pertinence'].mean():.2f}")
            logger.info(f"  Médiane: {df['score_pertinence'].median():.2f}")
            logger.info(f"  Écart-type: {df['score_pertinence'].std():.2f}")
            
            reader = Reader(rating_scale=(df['score_pertinence'].min(), 
                                       df['score_pertinence'].max()))
            
            self.data = Dataset.load_from_df(
                df[['id_utilisateur', 'titre_film', 'score_pertinence']], 
                reader
            )
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
            raise    
        finally:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Données chargées avec succès en {elapsed_time:.2f} secondes")
            
    def _log_progress(self, iteration: int, start_time: datetime) -> None:
        """Log la progression de la recherche sur grille."""
        elapsed_time = (datetime.now() - start_time).total_seconds()
        progress = (iteration + 1) / self.total_combinations
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time

        logger.info(f"Progression: {progress:.1%} ({iteration + 1}/{self.total_combinations})")
        logger.info(f"Temps écoulé: {elapsed_time:.1f}s")
        logger.info(f"Temps restant estimé: {remaining_time:.1f}s")

    def perform_gridsearch(self) -> None:
        """Exécute la recherche sur grille avec logs détaillés."""
        try:
            start_time = datetime.now()
            run_name = f"optimize-{start_time.strftime('%Y%m%d-%H%M%S')}"
            logger.info(f"Début de la recherche sur grille à {start_time.strftime('%H:%M:%S')}")
            
            # Récupérer l'URI de tracking MLflow depuis une variable d'environnement ou une connexion Airflow
            mlflow_tracking_uri = settings.MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Initialiser le client MLflow
            client = MlflowClient()

            experiment_name = "optimize_svd_surprise"

            # Vérifier si l'expérience existe et est active
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
    
            with mlflow.start_run(run_name=run_name):
                gs = GridSearchCV(
                    SVD,
                    self.param_grid,
                    measures=['rmse', 'mae'],
                    cv=5,
                    n_jobs=-1,
                    joblib_verbose=2  # Augmenté pour plus de détails
                )
            
                # Entraîner le modèle avec progression
                logger.info("Démarrage de l'entraînement...")
                gs.fit(self.data)

                # Enregistrer les meilleurs paramètres pour RMSE et MAE
                best_params_rmse = gs.best_params['rmse']
                best_score_rmse = gs.best_score['rmse']
                best_params_mae = gs.best_params['mae']
                best_score_mae = gs.best_score['mae']
                
                mlflow.log_params(best_params_rmse)
                mlflow.log_metric("best_rmse", best_score_rmse)
                mlflow.log_metric("best_mae", best_score_mae)

                # Enregistrer le fichier log comme artifact
                mlflow.log_artifact(str(file_path), artifact_path="log")

                logger.info("Optimisation terminée et enregistrée dans MLflow.")
            
            # Logger les résultats
            self.best_params_rmse = best_params_rmse
            self.best_params_mae = best_params_mae
            self.best_score_rmse = best_score_rmse
            self.best_score_mae = best_score_mae
            self.cv_results = gs.cv_results
            
            logger.info("\nRésultats de la recherche sur grille:")
            logger.info(f"Meilleurs paramètres (RMSE): {json.dumps(self.best_params_rmse, indent=2)}")
            logger.info(f"Meilleur score RMSE: {self.best_score_rmse:.4f}")
            logger.info(f"Meilleurs paramètres (MAE): {json.dumps(self.best_params_mae, indent=2)}")
            logger.info(f"Meilleur score MAE: {self.best_score_mae:.4f}")
            
            # Logger les résultats détaillés
            logger.info("\nTop 5 des meilleures combinaisons (RMSE):")
            sorted_results_rmse = sorted(
                zip(gs.cv_results['params'], gs.cv_results['mean_test_rmse']),
                key=lambda x: x[1]
            )
            for params, score in sorted_results_rmse[:5]:
                logger.info(f"RMSE: {score:.4f} avec {params}")
                
            logger.info("\nTop 5 des meilleures combinaisons (MAE):")
            sorted_results_mae = sorted(
                zip(gs.cv_results['params'], gs.cv_results['mean_test_mae']),
                key=lambda x: x[1]
            )
            for params, score in sorted_results_mae[:5]:
                logger.info(f"MAE: {score:.4f} avec {params}")
            
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            logger.info(f"\nRecherche sur grille terminée en {elapsed_time:.2f} secondes")
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche sur grille: {e}", exc_info=True)
            raise

    def save_results(self) -> None:
        """Sauvegarde les résultats au format JSON avec logs détaillés et les logs dans MLflow."""
        try:
            start_time = datetime.now()
            logger.info("Préparation des résultats pour la sauvegarde...")
            
            # Convertir les paramètres numpy en types Python natifs
            best_params_rmse = {k: int(v) if isinstance(v, np.integer) 
                            else float(v) if isinstance(v, np.floating)
                            else v 
                            for k, v in self.best_params_rmse.items()}
            
            best_params_mae = {k: int(v) if isinstance(v, np.integer)
                            else float(v) if isinstance(v, np.floating)
                            else v
                            for k, v in self.best_params_mae.items()}
            
            results = {
                "best_parameters": {
                    "rmse": best_params_rmse,
                    "mae": best_params_mae
                },
                "metrics": {
                    "rmse": float(self.best_score_rmse),
                    "mae": float(self.best_score_mae),
                    "timestamp": datetime.now().isoformat(),
                    "training_details": {
                        "total_combinations_tested": int(self.total_combinations),
                        "cross_validation_folds": 5,
                    }
                },
                "search_space": {k: list(map(float, v)) if isinstance(v[0], np.number) else v 
                            for k, v in self.param_grid.items()},
                "cross_validation_details": {
                    "folds": 5,
                    "all_scores": {
                        str(params): {
                            "rmse": float(rmse),
                            "mae": float(mae)
                        }
                        for params, rmse, mae in zip(
                            self.cv_results['params'],
                            self.cv_results['mean_test_rmse'],
                            self.cv_results['mean_test_mae']
                        )
                    }
                }
            }
            
            output_path = settings.OPTIMIZED_PARAMETERS_PATH
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            file_size = output_path.stat().st_size / 1024  # Taille en KB
            
            # Loguer les paramètres comme artifact dans MLflow
            mlflow.log_artifact(str(output_path), artifact_path="parameters")

            logger.info(f"Résultats sauvegardés dans {output_path}")
            logger.info(f"Taille du fichier: {file_size:.2f} KB")
            logger.info(f"Sauvegarde effectuée en {elapsed_time:.2f} secondes")
 
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {e}", exc_info=True)
            raise

def main():
    start_time = datetime.now()
    logger.info("=== Début de l'optimisation des paramètres SVD ===")
    logger.info("Version optimisée avec espace de recherche réduit")
    
    try:
        optimizer = SVDOptimizer()
        
        logger.info("\n1. Chargement des données...")
        optimizer.load_data()
        
        logger.info("\n2. Exécution de la recherche sur grille...")
        optimizer.perform_gridsearch()
        
        logger.info("\n3. Sauvegarde des résultats...")
        optimizer.save_results()
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n=== Optimisation terminée avec succès en {total_time:.2f} secondes ===")
        
    except Exception as e:
        logger.error(f"Erreur critique lors de l'optimisation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()