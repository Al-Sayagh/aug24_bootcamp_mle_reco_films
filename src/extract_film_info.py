from pathlib import Path
import sys
import logging
import os
import pandas as pd
from typing import Dict
import json

# Ajouter le répertoire racine au sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import local
from src.config import settings

# Configurer le logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_path = settings.FILM_EXTRACTOR_LOG_PATH
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

class FilmExtractor:
    def __init__(self):
        """
        Initialise l'extracteur d'informations sur les films.
        Utilise des chemins relatifs par rapport à la racine du projet.
        """
        self.input_path = settings.RAW_DATA_PATH
        self.output_path = settings.FILMS_INFO_PATH
        logger.info(f"Initialisation de l'extracteur de films avec {self.input_path}")

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

            df = pd.read_csv(self.input_path, encoding="utf-8")
            logger.info(f"Fichier lu avec succès - {len(df)} lignes")

            # Vérifier les colonnes minimales nécessaires
            required_columns = ['id_film', 'titre_film']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Colonnes manquantes dans le CSV: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            return df

        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier: {str(e)}", exc_info=True)
            raise

    def process_films(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Extrait la liste des films (id_film, titre_film),
        supprime les doublons, trie alphabétiquement et
        renvoie un dictionnaire {id_film: titre_film}.
        """
        try:
            logger.info("Début du traitement des données films")

            # Extraire les champs nécessaires
            films = df[['id_film', 'titre_film']].drop_duplicates()

            # Trier par ordre alphabétique du titre
            films = films.sort_values(by='titre_film')

            # Convertir en dictionnaire {id_film: titre_film}
            films_dict = films.set_index('id_film')['titre_film'].to_dict()

            logger.info(f"Traitement terminé - {len(films_dict)} films extraits")
            return films_dict

        except Exception as e:
            logger.error(f"Erreur lors du traitement des films: {str(e)}", exc_info=True)
            raise

    def save_json(self, films_dict: Dict[int, str]) -> None:
        """
        Sauvegarde les données de films au format JSON.
        Crée le répertoire de sortie si nécessaire.
        """
        try:
            logger.info(f"Sauvegarde des données dans {self.output_path}")

            # Créer le répertoire si nécessaire
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Écrire le dictionnaire dans un fichier JSON
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(films_dict, f, ensure_ascii=False, indent=4)

            file_size_kb = self.output_path.stat().st_size / 1024
            logger.info(f"Fichier JSON sauvegardé avec succès ({file_size_kb:.1f} KB)")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier: {str(e)}", exc_info=True)
            raise

    def run(self) -> None:
        """
        Exécute le processus complet d'extraction des informations de films.
        - Lit le CSV
        - Traite les données
        - Sauvegarde en JSON
        """
        try:
            logger.info("=== Début de l'extraction des données de films ===")
            df = self.read_data()
            films_dict = self.process_films(df)
            self.save_json(films_dict)
            logger.info("=== Extraction terminée avec succès ===")

        except Exception as e:
            logger.error("L'extraction des films a échoué", exc_info=True)
            sys.exit(1)

def main():
    """
    Point d'entrée principal du script.
    Instancie FilmExtractor et exécute le processus complet.
    """
    try:
        extractor = FilmExtractor()
        extractor.run()
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()