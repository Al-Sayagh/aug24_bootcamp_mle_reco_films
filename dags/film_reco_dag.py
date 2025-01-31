import sys
import os
import pandas as pd
import asyncio
import joblib
from surprise import Dataset, Reader
import logging
from datetime import datetime
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python_operator import PythonOperator

# Import local
from src.config import settings
from src.recommender import MovieRecommender
from src.extract_user_info import main as main_extract_user_info
from src.gridsearch import main as main_gridsearch

# Ajouter le PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Définir les fonctions utiles

    # Charger les données
def load_raw_data():
    """
    Lit le CSV original (via MovieRecommender) et enregistre le DataFrame tel quel dans data/processed/df_clean.csv.
    """
    recommender = MovieRecommender()
    recommender.load_csv()  
    df = recommender.df

    out_path = settings.RAW_DATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Fichier csv sauvegardé : {out_path}")

    # Préparer les données
def prepare_data_for_surprise():
    """
    Charge le CSV (df_clean.csv), appelle prepare_data() 
    pour générer le trainset, et sauvegarde le trainset dans trainset_surprise.pkl.
    """
    recommender = MovieRecommender()
    df_clean = pd.read_csv(settings.RAW_DATA_PATH)
    recommender.df = df_clean
    asyncio.run(recommender.prepare_data())
    print("Préparation Surprise terminée et trainset sauvegardé.")

    # Entraîner le modèle  
def train_model():
    """
    Charge le trainset sauvegardé, entraîne le modèle 
    (MovieRecommender.train()) et enregistre le modèle.
    """
    recommender = MovieRecommender()
    # Charger le trainset pkl depuis la tâche précédente
    recommender.trainset = joblib.load(settings.TRAINSET_PATH)

    # Entraîner le modèle (force=True pour forcer la ré-entrainement si besoin)
    asyncio.run(recommender.train(force=True))
    print("Modèle entraîné et sauvegardé.")

    # Evaluer le modèle  
def evaluate_model():
    """
    Charge le modèle et le trainset, évalue les performances (RMSE, MAE),
    puis enregistre les métriques sous forme de JSON (metrics/model_metrics.json).
    """
    recommender = MovieRecommender()
    recommender.trainset = joblib.load(settings.TRAINSET_PATH)
    # Recréer un dataset Surprise
    df_clean = pd.read_csv(settings.RAW_DATA_PATH)
    recommender.df = df_clean

    score_min = df_clean['score_pertinence'].min()
    score_max = df_clean['score_pertinence'].max()
    recommender.reader = Reader(rating_scale=(score_min, score_max))

    recommender.data = Dataset.load_from_df(
        recommender.df[['id_utilisateur', 'titre_film', 'score_pertinence']], 
        recommender.reader
        )
    
    recommender.model = joblib.load(settings.MODEL_PATH)
    
    metrics = asyncio.run(recommender.evaluate())
    print("Métriques d'évaluation calculés et sauvegardés :", metrics)

    # Obtenir des recommandations
def generate_recommendations():
    """
    Charge le modèle et les données, puis génère des recommandations pour un utilisateur donné.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    async def async_generate():
        try:
            recommender = MovieRecommender()

            # Charger le trainset
            trainset_path = settings.TRAINSET_PATH
            recommender.trainset = joblib.load(trainset_path)
            logger.info(f"Trainset chargé depuis {trainset_path}")

            # Charger le modèle
            model_path = settings.MODEL_PATH
            recommender.model = joblib.load(model_path)
            logger.info(f"Modèle chargé depuis {model_path}")

            # Charger le DataFrame
            df_path = settings.CLEAN_DATA_PATH
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
    dag_id='film_reco_pipeline',
    default_args=default_args,
    description='Pipeline complet pour la recommandation de films avec ML',
    schedule_interval='@once',
    catchup=False,
    tags=["projet"]
) as dag:

    # Groupe de tâches 1 : Charger et préparer les données
    with TaskGroup(group_id="data") as data_group:
        # Charger les données brutes
        load_raw_data_task = PythonOperator(
            task_id='load_raw_data',
            python_callable=load_raw_data,
        )
        # Préparer les données pour Surprise
        prepare_data_for_surprise_task = PythonOperator(
            task_id='prepare_data_for_surprise',
            python_callable=prepare_data_for_surprise,
            )
        # Extraire les informations utilisateurs
        extract_user_info_task = PythonOperator(
            task_id='extract_user_info',
            python_callable=main_extract_user_info,
            )
        # Orchestration au sein du group
        load_raw_data_task >> prepare_data_for_surprise_task
        load_raw_data_task >> extract_user_info_task

    # Groupe de tâches 2 : Entraîner, évaluer et optimiser le modèle
    with TaskGroup(group_id="model") as model_group:
        # Entraîner le modèle
        train_model_task = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
            )
        # Evaluer le modèle
        evaluate_model_task = PythonOperator(
            task_id='evaluate_model',
            python_callable=evaluate_model,
            )
        # Optimiser les hyperparamètres
        optimize_model_task = PythonOperator(
            task_id='optimize_model',
            python_callable=main_gridsearch,
            )
        
        # Orchestration au sein du groupe
        train_model_task >> evaluate_model_task
    
    # Groupe de tâches 3 : Générer les recommandations
    with TaskGroup(group_id="recommendations") as recommendations_group:    
        generate_recommendations_task = PythonOperator(
        task_id='generate_recommendations',
        python_callable=generate_recommendations,
    )
        
    # Groupe de tâches 4 : Mettre à jour les données et le modèle
    #with TaskGroup(group_id="refresh") as refresh_group:   
    
    # Orchestration générale
    prepare_data_for_surprise_task >> train_model_task >> generate_recommendations_task
    load_raw_data_task >> optimize_model_task
    