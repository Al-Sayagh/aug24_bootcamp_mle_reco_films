# Présentation des Données Brutes (INPUT) avec screenshot du CSV sur DVC

csv provenant du projet précédent regroupant des données de IMDb et MovieLens 
et un score de pertinence calculé;
    13 colonnes
        Utilisateur: ID#, Nom
        Film: ID#, Titre
            non utilisé par le modèle: Réalisateurs, Acteurs, Description
        Note: Score de pertinence (cible)
            non utilisé: Rating MovieLens, Rating MovieLens sur 10, Note moyenne IMDb, 
                Nombre de votes IMDb, Score de pertinence brute 

    407610 lignes (=évaluations)

    Nombre d'utilisateurs uniques: 13373
    Nombre de films uniques: 385
    Statistiques des scores de pertinence:
        Min: 0.00
        Max: 100.00
        Moyenne: 48.50
        Médiane: 47.15
        Écart-type: 21.46

# Présentation des Recommandations (OUTPUT) avec screenshot de l'api
- Choix d'un utilisateur
- Top X des films classées par score de pertinence parmi les films NON-VUS

# Présentation du Pipeline avec graph DagsHub + graph Airflow + screenshot web UI Airflow

    # Preprocessing (simplifié!)
    Trainset

    # Entraînement
    Modèle SVD Surprise

    # Evaluation
    Métriques
        rmse
        mae
        training_time
        nombre_utilisateurs
        nombre_films
        score_pertinence_moyen

    # Optimisation (Gridsearch)
    Hyperparamètres optimisés 
        n_factors
        n_epochs
        lr_all
        reg_all

    # Prédiction
    Recommandations 

# Présentation du Stack
- Github
    Structure
    Documentation
- DVC
    Dataframe
    Dagshub Data Pipeline
    Model
- Fast API
    (Root)
    Users
    Recommendations
    Gridsearch
    Refresh
- MLflow
    Experiences (train, evaluate, optimize)
    Runs
    Artifacts
- Airflow
    Pipeline complet (load, prepare, train, evaluate, optimize, predict)
    Graph View
- Docker
    Containers sur Docker Desktop (fastapi, airflow)

    
- (BentoML)
- (ZenML)
- (Prometheus & Grafana)


# Présentation de l'api
    (Root)
    Users
    Recommendations
    Gridsearch
    Refresh

# Présentation de MLflow 
    Experiences (train, evaluate, optimize)
    Runs
    Artifacts

# Présentation de Airflow

# Présentation de Docker

# Présentation de Bento ML

# Présentation de Zen ML

# Présentation de Prometheus & Grafana

# TABLEAU RECAPITULATIF

# Justification de nos choix
- Partir du modèle SVD Surprise
- Partir du dataset simplifié (et non des fichiers raw IMDb et MovieLens)
- Ne pas faire certaines chose: sécurisation de l'api, tests
- Justification du stack

# Difficultés rencontrées
- Difficultés techniques
    Problèmes d'installation et de compatibilité avec la bibliothèque Surprise
    Conflits entre différetns systèmes d'exploitation (Windows/Mac/Linux)
    Fiabilité de DVC
    Maîrise limitée de Python 
- Gestion du temps
    Difficulté à mener de front les cours et le projet (saut en difficulté vs formation Data Scientist)
    Trop peu de temps pour le projet
- Compréhension des concepts 
    Gestion de la machine hôte vs les conteneurs
    Système de recommandation à part dans la famille du Machine Learning
    Articulation entre les différents outils
        Redondances
        Quand faire le hand-off de l'un à l'autre?
        Comment les faire cohabiter?

# Pistes pour continuer le projet