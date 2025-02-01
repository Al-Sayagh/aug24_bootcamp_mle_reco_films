'''
TO DO / CLEANING
'''

# Github : DOCUMENTATION = ATTENDRE LUNDI ET MARDI
- Mettre à jour la structure finalisée dans le README principal (notamment effacer ou mettre à jour les WIP)
- Mettre à jour le fichier requirements.txt
- Créer un README d'instructions pour l'installation de tous les services (brouillon disponible == references/WIP_setup.md) 
- Ajouter un README pour chaque fichier avec du code py, yaml, dockerfile, etc. (demander à l'IA générative ce que fait chaque fichier en copiant/collant son contenu)
- [OPTIONNEL] Ajouter des liens utiles dans le document references/WIP_useful_links.md


. . .


# Modifier le recommender / script de prédiction
- Ajouter l'enregistrement du modèle via BentoML
- Ajouter un enregistrement des prédictions à la fin du recommender ?
    - JSON pour un utilisateur particulier (ou CSV avec les prédictions sur tout le dataframe, suivi par dvc) ?
- Ajouter le recommendations.json (ou predictions.csv) crée au schema DVC grâce au fichier dvc.yaml (mettre à  jour si besoin et dé-commenter)
- Ajouter une experiment "predict_svd_surprise" dans MLflow avec comme artefacts: recommender.log et le JSON de recommandations (ou CSV)

# Intégrer les prochains outils manquants
- Intégration Prometheus & Grafana 
- [OPTIONNEL] Intégration BentoML 
- [OPTIONNEL] Intégration ZenML 


. . .

# Autes outils manquants
- [ULTRA_OPTIONNEL] Alertes
- [ULTRA_OPTIONNEL] Ré-entraînement automatisé
- [ULTRA_OPTIONNEL] Scaling avec Kubernetes

# Airflow 
- [ULTRA_OPTIONNEL] Remplacer la tâche de recommandation par une tâche BentoML 
- [ULTRA_OPTIONNEL] Ajouter la tâche extract_film_info dans le DAG  => cela a aussi un impact sur le dvc.yaml
- [ULTRA_OPTIONNEL] Ajouter la tâche refresh_system

# FastAPI 
- [ULTRA_OPTIONNEL] Ajouter les Pytest, si possible pour chaque endpoint
- [ULTRA_OPTIONNEL] Ajouter une partie sécurisation à l'API (authentification, autorisation)
- [ULTRA_OPTIONNEL] Ajouter extract_film_info comme endpoint dans l'api (en plus du dvc.yaml et DAG)

# Code :
- [ULTRA_OPTIONNEL] Refactoriser le code pour séparer main et recommender en scripts spécialisés qui correspondent chacun à nos étapes de pipeline i.e. load, prepare, train, evaluate, optimize, predict

# Interface
- [ULTRA_OPTIONNEL] Montrer en plus des recommandations le top 10 des films préférés (déjà notés) et les recommandations hors des sentiers battus (random 5)
- [ULTRA_OPTIONNEL] Ajouter une UI user-friendly avec affiches de films, etc.