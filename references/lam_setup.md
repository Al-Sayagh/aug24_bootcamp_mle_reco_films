'''
CREER L'ENVIRONNEMENT
'''

# Cloner le repo github à partir de ce lien
https://github.com/Al-Sayagh/aug24_bootcamp_mle_reco_films.git

# Télécharger la base de données brutes (csv) ou alors par DVC une fois celui-ci établi
https://drive.google.com/file/d/1G6h50Pj-OsYL_S6GxCTy9PHThupnAyD2/view?usp=sharing

# Créer et accéder au dossier du projet et y copier le csv
cd /Users/lampaturle/Desktop/SWITCH_PROJECT/Datascientest/aug24_bootcamp_mle_reco_films/

# Créer un nouvel environnement virtuel 
mamba create -n mlopsproject python=3.9 -y

    # Effacer l'environnement virtuel 
    mamba remove -n mlopsproject --all

# Activer l'environnement virtuel 
mamba activate mlopsproject

    # Désactiver l'environnement virtuel
    mamba deactivate


'''
CONFIGURER LE DVC
'''

# Installer les librairies
pip install dvc
pip install "dvc[s3]"

# Initialiser le dvc
dvc init -f

# Configurer le DagsHub DVC remote
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/Al-Sayagh/aug24_bootcamp_mle_reco_films.s3

dvc remote modify origin --local access_key_id 8a08ba4d3a9988a7f9c1664a9c97b2354641694d
dvc remote modify origin --local secret_access_key 8a08ba4d3a9988a7f9c1664a9c97b2354641694d

dvc remote default origin

# Créer le fichier dvc.yaml
    # Ajout des stages selon le format suivant:

    stages:
  gridsearch:
    cmd: python src/models/grid_search.py
    deps:
    - data/processed/X_train_scaled.csv
    - data/processed/y_train.csv
    - src/models/grid_search.py
    - params.yaml
    outs:
    - models/best_params.pkl
    
# Ajouter le dataset sous versioning DVC
dvc add data/raw/df_demonstration.csv

# Faire un commit pour le suivi du dataset dans Git
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Ajout du dataset sous DVC"

# Faire un commit pour le suivi du modèle
git rm -r --cached 'models/svd_model.joblib'
git commit -m "stop tracking models/svd_model.joblib"

dvc add models/svd_model.joblib

git add models/svd_model.joblib.dvc .gitignore
git commit -m "Ajout du modèle sous DVC"

# Pousser les données vers le stockage distant (S3)
dvc push


'''
METTRE EN PLACE LE SYSTEME DE RECOMMANDATION
'''

# Installer les librairies
pip install cachetools
pip install scikit-surprise
pip install pandas
pip install pydantic-settings

# Gérer les conflits
pip install numpy==1.24.3 # Reinstall (i.e downgrade numpy)

# Configurer le chemin python (si nécessaire, aussi très utile pour le debugging):
export PYTHONPATH="/Users/lampaturle/Desktop/SWITCH_PROJECT/Datascientest/aug24_bootcamp_mle_reco_films:$PYTHONPATH"

# Lancer main.py pour debugging
python api/main.py


'''
CREER L'API
'''

# Installer les librairies
pip install fastapi
pip install uvicorn

# Lancer l'api
uvicorn app.main:app --reload

    # Arrêter l'api
    ^C

# Accéder à l'interface web
http://localhost:8000/docs


'''
DEFINIR LA SUITE DE TESTS
'''

# Installer les librairies
pip install pytest 

# Créer les fichiers test de sécurité
- conftest.py
- test_authentication.py
- test_authorization.py

# Créer les fichiers test de fonctionnalités
- test_get_recommendations.py
- test_get_metrics.py
- test_gridsearch.py
- test_get_users.py
- test_refresh_system.py

# Lancer les tests
pytest


'''
CONFIGURER ML FLOW
'''

# Installer les librairies
pip install mlflow

# Lancer gridsearch.py pour debugging
python src/gridsearch.py

# Configurer le serveur de tracking (en arrière plan)
mlflow server --host 0.0.0.0 --port 8081 &

    # Arrêter le serveur de tracking
    ^C

    # Arrêter les processus en cours sur le port 8081
    kill -9 $(lsof -t -i:8081)

    # Supprimer le dossier mlruns
    rm -r mlruns


'''
CONFIGURER AIRFLOW
'''

# Installer les librairies
pip install apache-airflow

# Gérer les conflits
apache-airflow-providers-openlineage>=1.8.0
email-validator>=2.0.0

# Télécharger le fichier docker-compose.yml 
wget https://dst-de.s3.eu-west-3.amazonaws.com/airflow_fr/eval/docker-compose.yaml

# Modifier le fichier pour adaptation au projet 
- Ajouter le Dockerfile en contexte
- Changer les volumes pour data/raw et data/processed et ajouter les volumes nécessaires en faisant bien attention aux chemins
- Eliminer la section dashboard
- Eliminer la section flower (dans l'intention d'utiliser Prometheus et Grafana)
- Changer le nom de l'image
- Ajouter le PYTHONPATH
- Enlever les exemples de dags dans l'interface web

# Créer le Dockerfile-airflow
- Changement de l'image pour python-3.9
- Workaround pour installation de surprise
- Installation des dépendances via requirements.txt
- Nettoyage du workaround pour optimiser la taille de l'image
- RUN pip install --no-cache-dir --upgrade "protobuf==4.21.12" en tant que user airflow (conflit)

# Créer les dossiers nécessaires et modifier les permissions
mkdir ./dags ./logs ./plugins ./metrics 

sudo chmod -R 777 dags/
sudo chmod -R 777 logs/
sudo chmod -R 777 plugins/
sudo chmod -R 777 metrics/

# Configurer les paramètres airflow
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

# Initialiser la base de données airflow (seulement la première fois)
docker compose up airflow-init 

docker compose build --no-cache

# Lancer les services airflow
docker compose up -d

    # Arrêter les services 
    docker compose down

# Accéder à l'interface web
http://localhost:8080

    # S'identifier sur airflow
    User: airflow
    Password: airflow

# Créer le DAG
- Définir le DAG
- Gérer les conflits
- Importer et modifier les fonctions
- Encadrer les fonctions asynchrones
- Définir les fonctions et les groupes
- Ajouter le PYTHONPATH

# Créer le fichier requirements.txt simplifié
pip install pipreqs
pipreqs ./ --force


'''
CONTENEURISER AVEC DOCKER
'''

# Modifier le fichier docker-compose
- Ajouter un network commun aux conteneurs qui doivent communiquer entre eux
- Ajouter le service mlflow sur le port 8081 avec une image mlflow créée à partir d'une image déjà existante et des volumes pour les métadonnées et les artifacts. Définir les paramètres backend-store-uri et default-artifact-root
- Ajouter le service fast api sur le port 8000 avec une image custom et en ajustant les dépendances. Utiliser un Dockerfile custom pour gérer surprise
- Modifier la structure du projet et la fonction main
- Ajouter les volumes nécessaires en vérifiant les chemins pour mlflow et fastapi
- Passer les identifiants en variables d'environnement .env.docker (attention au _ pour les credentials airflow)
- Ajouter des healthchecks, surtout à fast api
- Mettre airflow-init en premier dans l'ordre des services airflow
- Ajouter la MLFLOW_TRACKING_URI: http://mlflow:8081 à airflow et fastapi
- Gestion du warning git

# Ajuster les permissions des dossiers MLflow
sudo chmod -R 777 ./mlruns
sudo chmod -R 777 ./mlartifacts

# Changer le nom des images
docker tag mlflow:latest surprise-mlflow:latest
docker tag sha256:3315fb2b4701dac3fd50b6921df999fec2ee76e8bfcecdbc0daef096281b35ce surprise-airflow:2.8.1

# Vérifier ce qu'il se passe dans un conteneur
docker compose logs fastapi --follow

# Vérifier les ports
sudo lsof -i :8000
sudo lsof -i :8081


'''
DEPLOYER LE MODELE AVEC BENTOML
'''

# Installer les librairies
pip install bentoml

# Lancer le script train avec l'enregistrement du modèle dans bentoml models

# Vérifier l'enregistrement du modèle(tag="recommender_surpriseSVD:<tag>")
bentoml models list

# Charger le modèle
bentoml models get recommender_surpriseSVD:latest

# Changer de dossier
cd src

# Démarrer le service
bentoml serve service.py:recommender_service --reload

# Tester le service (autre console)
mamba activate mlopsproject
cd /Users/lampaturle/Desktop/SWITCH_PROJECT/Datascientest/aug24_bootcamp_mle_reco_films/
pytest tests/

# Créer le bento
bentoml build

# Vérifier la création du bento
bentoml list

# Créer l'image Docker
bentoml containerize recommender_service:latest

# Vérifier la création de l'image
docker images

# Tester l'image
docker run --rm -p 3000:3000 recommender_service:latest
docker run -it recommender_service:latest bash

# Compresser l'image Docker
docker save -o ./bento_image.tar recommender_service:latest

# Tester à nouveau le service
bentoml serve service.py:surpriseSVD_service --reload


'''
AJOUTER UNE COUCHE D'ABSTRACTION AVEC ZEN ML
'''

# Installer les librairies
pip install "zenml[templates,server]"

# Initialiser le projet
zenml init --template starter --template-with-defaults

# Démarrer le serveur
zenml up

    # Fermer le dashboard
    zenml down

# Pipeline
[Extraction des données] --> [Préparation des données] --> [Entraînement] --> [Évaluation] --> [Déploiement]



'''
METTRE EN PLACE UN DASHBOARD GRAFANA
'''


'''
CONFIGURER LES ALERTES
'''


'''
METTRE EN PLACE UN SYSTEME DE RE-ENTRAINEMENT AUTOMATISE
'''


'''
GERER LE SCALING AVEC KUBERNETES
'''






'''
CLEANING
'''

# Pres
- Demander à l"AI un tableau récapitulatif et comparatif de ce que font les différents outils dans notre projet

# Code
- Intégration BentoML
- Intégration ZenML ?
- Ajouter un enregistrement des prédictions à la fin du recommender ?
    - JSON pour un utilisateur particulier OU un CSV avec des prédictions sur tout le dataframe (suivi par dvc)

# Github : DOCUMENTATION = ATTENDRE LUNDI
- Mettre à jour le README principal avec la structure finalisée
- Créer un README d'instructions pour l'installation de tous les services 
- Ajouter un README pour tous les fichiers avec du code py, yaml, dockerfile, etc. (demander à l'AI ce que fait chaque fichier en copiant/collant son contenu)

# DVC
- Ajouter recommendations.json ou predictions.csv au shema grâce au fichier dvc.yaml (uncomment)

# FASTAPI : 
- Ajouter les Pytest, si possible pour chaque endpoint
- Ajouter une partie sécurisation à l'API (authentification, autorisation)
- Ajouter extract_film_info comme endpoint dans l'api (puis au dvc.yaml et au DAG)

# MLflow : ajouter une expérience "predict" 
- Experiment predict_svd_surprise
- Artefacts: recommender.log et JSON ou CSV de recommandations

# Airflow : 
- Remplacer la tâche de recommandation par une tâche BentoML ?
- Ajouter la tâche extract_film_info ?
- Ajouter la tâche refresh_system ?

# Code (optionnel) :
- Refactoriser le code pour le séparer en scripts qui correspondent exactement à nos étapes (load, prepare, train, evaluate, optimize, predict)
- Montrer en plus des recommandations les films préférés (déjà notés) et les recommandations hors des sentiers battus (random 5)