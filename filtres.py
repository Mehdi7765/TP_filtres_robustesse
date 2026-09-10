"""
FILTRES D'IMAGE À INTENSITÉ RÉGLABLE
====================================
Chaque filtre est une classe qui possède :
  - un nom, une intensité courante, ses bornes (minimum, maximum) et son pas ;
  - `augmenter()` / `diminuer()` qui modifient l'intensité SANS sortir des
    bornes (un flou avec un noyau pair ou nul ferait planter OpenCV) ;
  - `appliquer(image)` qui renvoie une NOUVELLE image, toujours en couleur
    (3 canaux BGR) et de la même taille que l'entrée. C'est ce qui permet de
    coller les deux images côte à côte et de les envoyer telles quelles au
    modèle : une image en niveaux de gris n'a qu'un canal, elle est donc
    reconvertie en BGR avant d'être renvoyée.

L'intensité n'a pas le même sens pour chaque filtre : le docstring de chaque
classe explique ce qu'elle représente. Dans tous les cas, « augmenter »
dégrade davantage l'image.

Les cinq filtres demandés : NiveauxDeGris, Flou, Contours, Seuillage, puis
les filtres personnels : Bruit, Pixellisation, Posterisation, Assombrissement,
Inversion.
"""
import cv2
import numpy as np


class Filtre:
    """Classe de base : gestion de l'intensité et de ses bornes."""
    nom = "identite"
    unite = ""          # texte affiché après la valeur (« px », « % », ...)
    minimum = 0
    maximum = 100
    pas = 1
    defaut = 0

    def __init__(self):
        self.intensite = self.defaut

    # --- réglage --------------------------------------------------------
    def augmenter(self):
        self.intensite = min(self.maximum, self.intensite + self.pas)

    def diminuer(self):
        self.intensite = max(self.minimum, self.intensite - self.pas)

    def reinitialiser(self):
        self.intensite = self.defaut

    def valeurs_possibles(self):
        """Toutes les intensités atteignables au clavier (pour l'étalonnage)."""
        return list(range(self.minimum, self.maximum + 1, self.pas))

    def texte_intensite(self):
        return f"{self.intensite}{self.unite}"

    # --- traitement -----------------------------------------------------
    def appliquer(self, image):
        return image.copy()


def gris_vers_bgr(gris):
    """Image 1 canal -> 3 canaux identiques (pour l'affichage et le modèle)."""
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
#  LES CINQ FILTRES DEMANDÉS
# ---------------------------------------------------------------------------
class NiveauxDeGris(Filtre):
    """Désaturation progressive.

    Un passage en niveaux de gris est tout ou rien. Pour le rendre progressif,
    l'intensité est le POURCENTAGE de désaturation : on mélange l'image
    couleur et sa version grise (0 % = couleur d'origine, 100 % = gris pur).
    On observe ainsi si le modèle a besoin de la couleur, et à partir de quel
    degré de désaturation il décroche (spoiler : presque jamais).
    """
    nom = "niveaux de gris"
    unite = " %"
    minimum, maximum, pas, defaut = 0, 100, 10, 100

    def appliquer(self, image):
        gris = gris_vers_bgr(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        alpha = self.intensite / 100.0
        return cv2.addWeighted(image, 1.0 - alpha, gris, alpha, 0)


class Flou(Filtre):
    """Flou gaussien. L'intensité est la taille du noyau en pixels.

    OpenCV exige un noyau IMPAIR et strictement positif : le minimum est 1
    (aucun flou) et le pas vaut 2, l'intensité reste donc toujours impaire.
    """
    nom = "flou"
    unite = " px"
    minimum, maximum, pas, defaut = 1, 61, 2, 5

    def appliquer(self, image):
        if self.intensite <= 1:
            return image.copy()
        k = self.intensite
        return cv2.GaussianBlur(image, (k, k), 0)


class Contours(Filtre):
    """Détecteur de contours de Canny : l'image devient un dessin au trait.

    L'intensité est le seuil haut de Canny (le seuil bas vaut la moitié).
    Plus il est élevé, moins il reste de contours : à 400 il ne reste presque
    plus rien. L'image ne contient plus que des traits blancs sur fond noir.
    """
    nom = "contours"
    unite = ""
    minimum, maximum, pas, defaut = 10, 400, 10, 100

    def texte_intensite(self):
        return f"seuil {self.intensite}"

    def appliquer(self, image):
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bords = cv2.Canny(gris, self.intensite / 2, self.intensite)
        return gris_vers_bgr(bords)


class Seuillage(Filtre):
    """Seuillage binaire : chaque pixel devient noir ou blanc.

    L'intensité est la valeur (0-255) qui sépare le noir du blanc. En dessous
    de ~40 presque tout est blanc, au-dessus de ~200 presque tout est noir.
    """
    nom = "seuillage"
    unite = ""
    minimum, maximum, pas, defaut = 0, 255, 5, 128

    def texte_intensite(self):
        return f"seuil {self.intensite}/255"

    def appliquer(self, image):
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binaire = cv2.threshold(gris, self.intensite, 255, cv2.THRESH_BINARY)
        return gris_vers_bgr(binaire)


# ---------------------------------------------------------------------------
#  FILTRES PERSONNELS
# ---------------------------------------------------------------------------
class Bruit(Filtre):
    """Bruit gaussien additif. L'intensité est l'écart-type du bruit (en
    niveaux de gris sur 255).

    Un bruit de sigma 20 est à peine visible à l'œil ; le modèle, lui, y est
    très sensible : c'est le filtre le plus instructif du TP (voir
    docs/observations.md). Le bruit est tiré à chaque image, il « grouille »
    à l'écran comme sur une vieille télévision.
    """
    nom = "bruit"
    unite = ""
    minimum, maximum, pas, defaut = 0, 120, 5, 20

    def __init__(self):
        super().__init__()
        self._generateur = np.random.default_rng()

    def texte_intensite(self):
        return f"sigma {self.intensite}"

    def appliquer(self, image):
        if self.intensite <= 0:
            return image.copy()
        bruit = self._generateur.normal(0.0, self.intensite, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + bruit, 0, 255).astype(np.uint8)


class Pixellisation(Filtre):
    """Gros pixels : on réduit l'image puis on l'agrandit sans interpolation.

    L'intensité est la taille d'un « gros pixel » en pixels réels. C'est une
    façon de simuler une caméra de très faible résolution : à 16 px, une image
    640x480 ne contient plus que 40x30 informations distinctes.
    """
    nom = "pixellisation"
    unite = " px"
    minimum, maximum, pas, defaut = 1, 64, 1, 8

    def appliquer(self, image):
        if self.intensite <= 1:
            return image.copy()
        h, l = image.shape[:2]
        petite = cv2.resize(image, (max(1, l // self.intensite), max(1, h // self.intensite)),
                            interpolation=cv2.INTER_AREA)
        return cv2.resize(petite, (l, h), interpolation=cv2.INTER_NEAREST)


class Posterisation(Filtre):
    """Réduction du nombre de couleurs. L'intensité est le nombre de bits
    supprimés sur chaque canal : 0 = 256 niveaux (image intacte),
    7 = 2 niveaux par canal, soit 8 couleurs possibles en tout.
    """
    nom = "posterisation"
    minimum, maximum, pas, defaut = 0, 7, 1, 5

    def texte_intensite(self):
        niveaux = 2 ** (8 - self.intensite)
        return f"{self.intensite} bits ({niveaux} niv./canal)"

    def appliquer(self, image):
        if self.intensite <= 0:
            return image.copy()
        # Met à zéro les bits de poids faible de chaque canal
        masque = np.uint8((0xFF << self.intensite) & 0xFF)
        return image & masque


class Assombrissement(Filtre):
    """Baisse de luminosité. L'intensité est le pourcentage de lumière retirée
    (100 % = image entièrement noire). Répond à la question « et si l'image
    est très sombre ? ».
    """
    nom = "assombrissement"
    unite = " %"
    minimum, maximum, pas, defaut = 0, 100, 5, 50

    def appliquer(self, image):
        facteur = 1.0 - self.intensite / 100.0
        return cv2.convertScaleAbs(image, alpha=facteur, beta=0)


class Inversion(Filtre):
    """Négatif progressif : mélange entre l'image et son négatif.
    L'intensité est le pourcentage d'inversion (50 % = gris uniforme,
    100 % = négatif complet). À l'œil, un négatif reste parfaitement
    lisible ; pour le modèle, un ciel noir et une ombre blanche ne
    ressemblent à rien de ce qu'il a appris.
    """
    nom = "inversion"
    unite = " %"
    minimum, maximum, pas, defaut = 0, 100, 10, 100

    def appliquer(self, image):
        negatif = cv2.bitwise_not(image)
        alpha = self.intensite / 100.0
        return cv2.addWeighted(image, 1.0 - alpha, negatif, alpha, 0)


# ---------------------------------------------------------------------------
def creer_filtres():
    """Liste des filtres dans l'ordre des touches 1..9."""
    return [NiveauxDeGris(), Flou(), Contours(), Seuillage(),
            Bruit(), Pixellisation(), Posterisation(), Assombrissement(), Inversion()]
