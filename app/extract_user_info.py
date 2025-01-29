import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import json
import sys

# Import local
from app.config import settings

# Configurer le logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_path = settings.USER_EXTRACTOR_LOG_PATH
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

class UserExtractor:
    def __init__(self):
        """
        Initialise l'extracteur d'informations utilisateurs.
        Utilise des chemins relatifs par rapport à la racine du projet.
        """
        self.input_path = settings.RAW_DATA_PATH
        self.output_path = settings.USERS_INFO_PATH
        logger.info(f"Initialisation de l'extracteur avec {self.input_path}")

    def read_data(self) -> pd.DataFrame:
        """
        Lit le fichier CSV et retourne un DataFrame.
        Vérifie l'existence du fichier et gère les erreurs.
        """
        try:
            logger.info(f"Lecture du fichier {self.input_path}")
            
            if not self.input_path.exists():
                error_msg = f"Fichier non trouvé: {self.input_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            df = pd.read_csv(self.input_path)
            logger.info(f"Fichier lu avec succès - {len(df)} lignes")
            
            # Vérifier les colonnes requises
            required_columns = ['id_utilisateur', 'nom_utilisateur', 'titre_film', 'note_utilisateur']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                error_msg = f"Colonnes manquantes dans le CSV: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier: {str(e)}", exc_info=True)
            raise

    def process_users(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Traite les données pour extraire les informations par utilisateur.
        Calcule les statistiques de visionnage et de notation.
        """
        try:
            logger.info("Début du traitement des données utilisateurs")
            
            # Calculer les statistiques par utilisateur
            user_stats = df.groupby(['id_utilisateur', 'nom_utilisateur']).agg({
                'titre_film': 'count',
                'note_utilisateur': ['mean', 'min', 'max']
            }).round(2)
            
            logger.info(f"Statistiques calculées pour {len(user_stats)} utilisateurs")
            
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "utilisateurs": []
            }
            
            # Pour chaque utilisateur
            for (user_id, user_name) in user_stats.index:
                user_info = {
                    "id": str(user_id),
                    "nom": user_name,
                    "nb_films_notes": int(user_stats.loc[(user_id, user_name), ('titre_film', 'count')]),
                    "notes": {
                        "moyenne": float(user_stats.loc[(user_id, user_name), ('note_utilisateur', 'mean')]),
                        "minimum": float(user_stats.loc[(user_id, user_name), ('note_utilisateur', 'min')]),
                        "maximum": float(user_stats.loc[(user_id, user_name), ('note_utilisateur', 'max')])
                    }
                }
                result["utilisateurs"].append(user_info)
            
            logger.info(f"Traitement terminé - {len(result['utilisateurs'])} utilisateurs traités")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données: {str(e)}", exc_info=True)
            raise

    def save_json(self, data: Dict[str, Any]) -> None:
        """
        Sauvegarde les données au format JSON.
        Crée les répertoires nécessaires si besoin.
        """
        try:
            logger.info(f"Sauvegarde des données dans {self.output_path}")
            
            # Créer le répertoire si nécessaire
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le fichier JSON
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            file_size = self.output_path.stat().st_size / 1024  # Taille en KB
            logger.info(f"Fichier JSON sauvegardé avec succès ({file_size:.1f} KB)")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier: {str(e)}", exc_info=True)
            raise

    def run(self) -> None:
        """
        Exécute le processus complet d'extraction.
        Gère les erreurs et assure une sortie propre.
        """
        try:
            logger.info("=== Début de l'extraction des données ===")
            df = self.read_data()
            processed_data = self.process_users(df)
            self.save_json(processed_data)
            logger.info("=== Extraction terminée avec succès ===")
            
        except Exception as e:
            logger.error("L'extraction a échoué", exc_info=True)
            sys.exit(1)

def main():
    """Point d'entrée principal du script."""
    try:
        extractor = UserExtractor()
        extractor.run()
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()