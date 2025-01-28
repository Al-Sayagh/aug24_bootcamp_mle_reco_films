from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import asyncio
import pandas as pd
from surprise import Reader, Dataset
import joblib
import sys
import os
import logging

# Import local
from app.config import settings
from app.recommender import MovieRecommender

# Ajouter le PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_recommendations():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    async def async_generate():
        try:
            recommender = MovieRecommender()

            # Charger le trainset
            trainset_path = "/data/processed/trainset_surprise.pkl"
            recommender.trainset = joblib.load(trainset_path)
            logger.info(f"Trainset chargé depuis {trainset_path}")

            # Charger le modèle
            model_path = settings.MODEL_PATH
            recommender.model = joblib.load(model_path)
            logger.info(f"Modèle chargé depuis {model_path}")

            # Charger le DataFrame
            df_path = "/data/processed/df_clean.csv"
            recommender.df = pd.read_csv(df_path)
            logger.info(f"DataFrame chargé depuis {df_path} avec {len(recommender.df)} lignes.")

            # Initialiser 'reader' et 'data'
            recommender.reader = Reader(rating_scale=(recommender.df['score_pertinence'].min(), recommender.df['score_pertinence'].max()))
            recommender.data = Dataset.load_from_df(
                recommender.df[['id_utilisateur', 'titre_film', 'score_pertinence']],
                recommender.reader
            )
            logger.info("Reader et Dataset initialisés.")

            # Créer le mapping des utilisateurs
            recommender.user_mapping = {str(uid): True for uid in recommender.df['id_utilisateur'].unique()}
            logger.info(f"Mapping créé pour {len(recommender.user_mapping)} utilisateurs uniques.")

            # Vérifier si une mise à jour est nécessaire
            if recommender.needs_update():
                logger.info("Mise à jour nécessaire, appel de load_csv() et train()")
                recommender.load_csv()
                await recommender.train(force=True)
            else:
                logger.info("Mise à jour non nécessaire, pas de réentraînement.")

            # Générer les recommandations
            recommendations = await recommender.recommend(id_utilisateur='10005', n=5)
            logger.info(f"Recommandations générées pour l'utilisateur 10005 : {recommendations}")

            # Afficher les recommandations
            if recommendations:
                print("Recommandations pour l'utilisateur 10005 :")
                for r in recommendations:
                    print(r)
            else:
                print("Aucune recommandation trouvée ou utilisateur inconnu.")

            # Optionnel : Retourner une valeur sérialisable
            return "Recommandations générées avec succès"

        except Exception as e:
            logger.error(f"Erreur lors de la génération des recommandations: {e}", exc_info=True)
            raise

    # Exécuter la coroutine et capturer le résultat
    result = asyncio.run(async_generate())

    # Optionnellement, retourner une valeur sérialisable
    return result

# Définir les arguments par défaut pour le DAG
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 28),
}

# Définir le DAG
with DAG(
    dag_id='test_pipeline',
    default_args=default_args,
    description='Pipeline de test',
    schedule_interval='@once',
    catchup=False,
    tags=["projet"]
) as dag:

    generate_recommendations_task = PythonOperator(
        task_id='generate_recommendations',
        python_callable=generate_recommendations,
    )
