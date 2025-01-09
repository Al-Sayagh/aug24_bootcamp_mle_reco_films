import pandas as pd
import json
import os

# Chemin relatif vers le fichier CSV (par rapport à la racine du projet)
chemin_fichier_csv = os.path.join("data", "raw", "df_demonstration.csv")

# Chemin relatif vers le fichier JSON de sortie
chemin_fichier_json = os.path.join("data", "processed", "films.json")

try:
    # Charger le DataFrame
    df = pd.read_csv(chemin_fichier_csv, encoding="utf-8")

    # Extraire les titres de films et leurs IDs
    films = df[['id_film', 'titre_film']].drop_duplicates()

    # Trier les films par titre alphabétiquement
    films = films.sort_values(by='titre_film')

    # Convertir en dictionnaire pour le format JSON
    films_dict = films.set_index('id_film')['titre_film'].to_dict()

    # Créer le dossier 'processed' s'il n'existe pas
    os.makedirs(os.path.dirname(chemin_fichier_json), exist_ok=True)

    # Enregistrer dans un fichier JSON
    with open(chemin_fichier_json, 'w', encoding="utf-8") as f:
        json.dump(films_dict, f, ensure_ascii=False, indent=4)

    print(f"Informations sur les films extraites et enregistrées dans {chemin_fichier_json}")

except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_fichier_csv} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")