Système de Recommandation de Films
==============================

Ce repo est le repo de notre projet ML Ops "Système de recommandation de films". 

Structure du Projet
------------

   
    ├── api                <- Fichiers pour l'api FastAPI.
    │       
    ├── dags               <- Fichier DAG pour Airflow. 
    │
    ├── data               <- Doit être sur votre ordinateur mais pas sur Github (seulement dans .gitignore).
    │   ├── processed      <- Data sets finaux et canoniques à utiliser pour la modélisation.
    │   └── raw            <- Data dump originel et immutable (non modifiable).
    │
    ├── logs               <- Fichiers de logging. 
    │                    
    ├── metrics            <- Sauvegarde des métriques du modèle. 
    │
    ├── mlartifacts        <- Artifacts MLflow. 
    │    
    ├── mlruns             <- Métadonnées ML Flow. 
    │ 
    ├── models             <- Sauvegarde du modèle. 
    │
    ├── notebooks          <- Notebooks Jupyter, scripts, rapport d'analyse exploratoire des données.
    │
    ├── references         <- Dictionnaires, manuels, liens, et autres documents explicatifs.
    │
    ├── src                <- Code source à utiliser pour ce projet.
    │ 
    ├── LICENSE            
    ├── README.md          <- README pour la structure du projet.
    │    
    ├── requirements.txt   <- Fichier requirements pour reproduire l'environnement.
    │   

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
