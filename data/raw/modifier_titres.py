import pandas as pd

def modifier_titres(df):
    """
    Modifie les titres de films dans la colonne 'titre_film' du DataFrame
    pour déplacer ", The" à la fin du titre au début.

    Args:
        df: DataFrame pandas contenant la colonne 'titre_film'.

    Returns:
        DataFrame pandas avec les titres modifiés dans la colonne 'titre_film'.
    """

    def deplacer_the(titre):
        if titre.endswith(", The"):
            return "The " + titre[:-5]
        else:
            return titre

    df['titre_film'] = df['titre_film'].apply(deplacer_the)  # Modification ici : 'titre_film' au lieu de 'title_fr'
    return df

# Charger le DataFrame depuis le fichier CSV
try:
    df_demonstration = pd.read_csv("df_demonstration.csv")
except FileNotFoundError:
    print("Erreur : Le fichier df_demonstration.csv n'a pas été trouvé.")
    exit()

# Appliquer la modification des titres
df_demonstration = modifier_titres(df_demonstration)

# Enregistrer le DataFrame modifié dans un nouveau fichier CSV
df_demonstration.to_csv("df_demonstration_modifie.csv", index=False)

print("Les titres ont été modifiés et enregistrés dans df_demonstration_modifie.csv")