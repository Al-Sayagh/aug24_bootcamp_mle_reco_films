import json
import logging
from pathlib import Path
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import itertools
import time

# Configuration du logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='gridsearch.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

# Ajout d'un handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class SVDOptimizer:
    def __init__(self):
        # Grille optimisée basée sur les meilleurs résultats précédents
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
        
        # Calcul du nombre total de combinaisons
        self.total_combinations = np.prod([len(values) for values in self.param_grid.values()])
        logger.info(f"Nombre total de combinaisons à tester: {self.total_combinations}")
        
        # Log des paramètres de recherche
        logger.info("Espace de recherche des paramètres (optimisé):")
        for param, values in self.param_grid.items():
            logger.info(f"  {param}: {values}")

    def load_data(self) -> None:
        """Charge les données depuis le CSV."""
        try:
            start_time = time.time()
            logger.info("Début du chargement des données...")
            
            df = pd.read_csv('data/raw/df_demonstration.csv', dtype={'id_utilisateur': str})
            
            logger.info(f"Dimensions du DataFrame: {df.shape}")
            logger.info(f"Nombre d'utilisateurs uniques: {df['id_utilisateur'].nunique()}")
            logger.info(f"Nombre de films uniques: {df['titre_film'].nunique()}")
            
            # Statistiques sur les scores de pertinence
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
            
            elapsed_time = time.time() - start_time
            logger.info(f"Données chargées avec succès en {elapsed_time:.2f} secondes")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}", exc_info=True)
            raise

    def _log_progress(self, iteration: int, start_time: float) -> None:
        """Log la progression de la recherche sur grille."""
        elapsed_time = time.time() - start_time
        progress = (iteration + 1) / self.total_combinations
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time

        logger.info(f"Progression: {progress:.1%} ({iteration + 1}/{self.total_combinations})")
        logger.info(f"Temps écoulé: {elapsed_time:.1f}s")
        logger.info(f"Temps restant estimé: {remaining_time:.1f}s")

    def perform_gridsearch(self) -> None:
        """Exécute la recherche sur grille avec logs détaillés."""
        try:
            start_time = time.time()
            logger.info(f"Début de la recherche sur grille à {datetime.now().strftime('%H:%M:%S')}")
            
            # Création du GridSearchCV avec un callback pour le logging
            gs = GridSearchCV(
                SVD,
                self.param_grid,
                measures=['rmse', 'mae'],
                cv=5,
                n_jobs=-1,
                joblib_verbose=2  # Augmenté pour plus de détails
            )
            
            # Fit avec progression
            logger.info("Démarrage de l'entraînement...")
            gs.fit(self.data)
            
            # Logging des résultats
            self.best_params_rmse = gs.best_params['rmse']
            self.best_params_mae = gs.best_params['mae']
            self.best_score_rmse = gs.best_score['rmse']
            self.best_score_mae = gs.best_score['mae']
            self.cv_results = gs.cv_results
            
            logger.info("\nRésultats de la recherche sur grille:")
            logger.info(f"Meilleurs paramètres (RMSE): {json.dumps(self.best_params_rmse, indent=2)}")
            logger.info(f"Meilleur score RMSE: {self.best_score_rmse:.4f}")
            logger.info(f"Meilleurs paramètres (MAE): {json.dumps(self.best_params_mae, indent=2)}")
            logger.info(f"Meilleur score MAE: {self.best_score_mae:.4f}")
            
            # Log des résultats détaillés
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
            
            elapsed_time = time.time() - start_time
            logger.info(f"\nRecherche sur grille terminée en {elapsed_time:.2f} secondes")
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche sur grille: {e}", exc_info=True)
            raise

    def save_results(self) -> None:
        """Sauvegarde les résultats au format JSON avec logs détaillés."""
        try:
            start_time = time.time()
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
            
            output_path = Path('data/processed/svd_optimization.json')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            elapsed_time = time.time() - start_time
            file_size = output_path.stat().st_size / 1024  # Taille en KB
            
            logger.info(f"Résultats sauvegardés dans {output_path}")
            logger.info(f"Taille du fichier: {file_size:.2f} KB")
            logger.info(f"Sauvegarde effectuée en {elapsed_time:.2f} secondes")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des résultats: {e}", exc_info=True)
            raise

def main():
    start_time = time.time()
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
        
        total_time = time.time() - start_time
        logger.info(f"\n=== Optimisation terminée avec succès en {total_time:.2f} secondes ===")
        
    except Exception as e:
        logger.error(f"Erreur critique lors de l'optimisation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()