'''
CREER L'ENVIRONNEMENT
'''

# Lien pour clonage du repo github
https://github.com/Al-Sayagh/aug24_bootcamp_mle_reco_films.git

# Chemin vers le dossier du projet
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
dvc remote modify origin endpointurl https://dagshub.com/LamPaturle-AI/examen-dvc.s3

dvc remote modify origin --local access_key_id 8a08ba4d3a9988a7f9c1664a9c97b2354641694d
dvc remote modify origin --local secret_access_key 8a08ba4d3a9988a7f9c1664a9c97b2354641694d

dvc remote default origin

# Créer le fichier paramètres
touch params.yaml

# Push vers le DVC
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

# Configurer le chemin python (si nécessaire):
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

# Créer les fichiers _test.py

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
- Changer le nom de l'image
- Ajouter le PYTHONPATH
- Enelever les exemples dans l'interface web

# Créer le Dockerfile-airflow
- Changement de l'image
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
docker compose up airflow-init

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

# Créer le fichier requirements.txt
pip install pipreqs
pipreqs ./ --force



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
BENTOML
'''

# Installer les librairies
pip install bentoml

# Lancer les scripts et l'enregistrement du modèle dans bentoml models

# Vérifier l'enregistrement du modèle(tag="recommendations_surpriseSVD:<tag>")
bentoml models list

# Charger le modèle
bentoml models get recommendations_surpriseSVD:latest

# Changer de dossier
cd src

# Démarrer le service
bentoml serve service.py:surpriseSVD_service --reload

# Tester le service (autre console)
mamba activate mlopsproject
cd /Users/lampaturle/Desktop/SWITCH_PROJECT/Datascientest/aug24_bootcamp_mle_reco_films/
pytest tests/

# Créer le bento
bentoml build

# Vérifier la création du bento
bentoml list

# Créer l'image Docker
bentoml containerize surpriseSVD_service:yphqrhgvrwlgfvrd -t surpriseSVD_service:latest

# Vérifier la création de l'image
docker images

# Tester l'image
docker run --rm -p 3000:3000 surpriseSVD_service:latest
docker run -it surpriseSVD_service:latest bash

# Compresser l'image Docker
docker save -o ./bento_image.tar surpriseSVD_service:latest

# Tester à nouveau le service
bentoml serve service.py:surpriseSVD_service --reload



'''
CLEANING
'''

# Ajouter une partie sécurisation à l'API (authentification, autorisation)

# Ajouter extract_film_info
- comme endpoint dans l'api
- comme task dans Airflow 

# Montrer aussi les films préférés (déjà notés) et les recommandations hors des sentiers battus

# Intégrer Zen ML