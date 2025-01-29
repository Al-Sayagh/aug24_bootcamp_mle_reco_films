Système de Recommandation de Films
==============================

Ce repo est le repo de notre projet ML Ops "Système de recommandation de films". 

Structure du Projet
------------

    ├── LICENSE            
    ├── README.md          <- README principal pour les développeurs de ce projet.
    │
    ├── app                <- Code source à utiliser pour ce projet.
    │
    ├── dags               <- Fichier DAG pour Airflow. 
    │
    ├── data               <- Doit être sur votre ordinateur mais pas sur Github (seulement dans .gitignore).
    │   ├── processed      <- Data sets finaux et canoniques à utiliser pour la modélisation.
    │   └── raw            <- Data dump originel et immutable (non modifiable).
    │                         
    ├── metrics            <- Sauvegarde des métriques du modèle. 
    │ 
    ├── models             <- Sauvegarde du modèle. 
    │
    ├── notebooks          <- Notebooks Jupyter, scripts, rapport d'analyse exploratoire des données.
    │
    ├── references         <- Dictionnaires, manuels, liens, et autres documents explicatifs.
    │
    ├── tests              <- Fichiers test pour Pytest. 
    │
    ├── requirements.txt   <- Fichier requirements pour reproduire l'environnement d'analyse.
    │   

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
