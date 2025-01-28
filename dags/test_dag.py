from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import asyncio
import pandas as pd
import joblib
import sys
import os

# Import local
from app.config import settings
from app.recommender import MovieRecommender

# Ajouter le PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_recommendations():
    """
    Charge le modèle et les données, puis génère des recommandations pour un utilisateur donné.
    """
    recommender = MovieRecommender()
    recommender.trainset = joblib.load("/data/processed/trainset_surprise.pkl")
    recommender.model = joblib.load(settings.MODEL_PATH)  
    recommender.df = pd.read_csv("/data/processed/df_clean.csv")
    
    # Par exemple pour l'utilisateur 10005, on demande 5 recommandations
    recommendations = asyncio.run(recommender.recommend(id_utilisateur=10005, n=5))
    
    if recommendations:
        print("Recommandations pour l'utilisateur 10005 :")
        for r in recommendations:
            print(r)
    else:
        print("Aucune recommandation trouvée ou utilisateur inconnu.")

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
