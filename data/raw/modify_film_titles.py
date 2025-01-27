import pandas as pd

def modify_film_titles(df):
    """
    Modifie les titres de films dans la colonne 'titre_film' du DataFrame
    pour déplacer ", The" à la fin du titre au début.

    Args:
        df: DataFrame pandas contenant la colonne 'titre_film'.

    Returns:
        DataFrame pandas avec les titres modifiés dans la colonne 'titre_film'.
    """

    def move_the(title):
        if title.endswith(", The"):
            return "The " + title[:-5]
        else:
            return title

    df['titre_film'] = df['titre_film'].apply(move_the)
    return df

# Charger le DataFrame depuis le fichier CSV
try:
    df_demonstration = pd.read_csv("data/raw/df_demonstration.csv")
except FileNotFoundError:
    print("Erreur : Le fichier df_demonstration.csv n'a pas été trouvé.")
    exit()

# Appliquer la modification des titres
df_demonstration = modify_film_titles(df_demonstration)

# Enregistrer le DataFrame modifié dans un nouveau fichier CSV
df_demonstration.to_csv("data/processed/df_demonstration_modified.csv", index=False)

print("Les titres ont été modifiés et enregistrés dans df_demonstration_modified.csv")