'''
CREER L'ENVIRONNEMENT
'''

# Cloner le repo github à partir de ce lien
https://github.com/Al-Sayagh/aug24_bootcamp_mle_reco_films.git

# Télécharger la base de données brutes (csv)
https://drive.google.com/file/d/1G6h50Pj-OsYL_S6GxCTy9PHThupnAyD2/view?usp=sharing

# Accéder au dossier du projet et y copier le csv
/Users/lampaturle/Desktop/SWITCH_PROJECT/Datascientest/aug24_bootcamp_mle_reco_films/

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
    params:
    - model.n_estimators
    - model.max_depth
    - model.learning_rate
  train:
    cmd: python src/models/training.py
    deps:
    - data/processed/X_train_scaled.csv
    - data/processed/y_train.csv
    - models/best_params.pkl
    - src/models/training.py
    outs:
    - models/gbr_model.pkl

# Ajouter le dataset sous versioning DVC
dvc add data/raw/df_demonstration.csv

# Commit du suivi du dataset dans Git
git add data/raw/dataset.csv.dvc .gitignore
git commit -m "Ajout du dataset sous DVC"

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

# Configurer le chemin python (si nécessaire, aussi très utile pour le debug):
export PYTHONPATH="/Users/lampaturle/Desktop/SWITCH_PROJECT/Datascientest/aug24_bootcamp_mle_reco_films:$PYTHONPATH"

# Lancer main.py pour debug
python app/main.py


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

# Lancer gridsearch.py pour debug
python app/gridsearch.py

# Configurer le serveur de tracking
mlflow server --host 0.0.0.0 --port 8081

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
- Changer les volumes pour data/raw et data/processed et ajouter les volumes nécessaires
- Eliminer la section dashboard
- Eliminer la section flower (dans l'intention d'utiliser Prometheus et Grafana)
- Changer le nom de l'image
- Ajouter le PYTHONPATH
- Enlever les exemples dans l'interface web

# Créer le Dockerfile-airflow
- Changement de l'image pour python-3.9
- Workaround pour installation de surprise
- Installation des dépendances via requirements.txt
- Nettoyage pour optimiser la taille de l'image

# Créer les dossiers nécessaires et modifier les permissions
mkdir ./dags ./logs ./plugins ./metrics 

sudo chmod -R 777 dags/
sudo chmod -R 777 logs/
sudo chmod -R 777 plugins/
sudo chmod -R 777 metrics/

# Configurer les paramètres airflow
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > .env

# Initialiser airflow
docker compose build --no-cache
docker compose up airflow-init (seulement la première fois)

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
CONTENAIRISER AVEC DOCKER
'''

# Modifier le fichier docker-compose
- Ajouter le service mlflow sur le port 8081 avec des volumes pour les métadonnées et les artifacts
- Ajouter le service fast api sur le port 8000 en ajustant les dépendances
- Passer les identifiants en variables d'environnement
- Ajouter un mot de passe Redis
- Ajouter des healthchecks à mlflow et fast api
- Mettre airflow-init en premier


# Ajuster les permissions des dossiers MLflow
sudo chown -R 500:500 ./mlruns ./mlartifacts


'''
DEPLOYER AVEC BENTOML
'''

# Installer les librairies
pip install bentoml

# Lancer les scripts et l'enregistrement du modèle dans bentoml models

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
METTRE EN PLACE UN DASHBOARD GRAFANA
'''


'''
CONFIGURER LES ALERTES
'''


'''
SYSTEME DE RE-ENTRAINEMENT AUTOMATISE
'''


'''
KUBERNETES
'''






'''
CLEANING
'''

# Ajouter l'expérience predict à MLflow == get recommendations (artefacts: log et JSON de recommandations)

# Ajouter une partie sécurisation à l'API (authentification, autorisation)

# Ajouter extract_film_info
- comme endpoint dans l'api
- comme task dans Airflow 

# Montrer aussi les films préférés (déjà notés) et les recommandations hors des sentiers battus

# Intégrer Zen ML