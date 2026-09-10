"""
CONFIGURATION DU TP
===================
Tous les réglages sont regroupés ici. Chaque script accepte aussi des options
en ligne de commande (--camera, --taille-entree, ...) qui prennent le dessus
sur ces valeurs, pour changer un paramètre en séance sans toucher au fichier.
"""
import os

DOSSIER_PROJET = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
#  CAMÉRA
# ---------------------------------------------------------------------------
INDEX_CAMERA = 0          # 0 = première webcam détectée
LARGEUR_CAPTURE = 640     # résolution demandée à la caméra (elle peut refuser)
HAUTEUR_CAPTURE = 480

# ---------------------------------------------------------------------------
#  MODÈLE (YOLOv4-tiny au format Darknet, lu par le module DNN d'OpenCV)
# ---------------------------------------------------------------------------
DOSSIER_MODELES = os.path.join(DOSSIER_PROJET, "modeles")
FICHIER_CFG = os.path.join(DOSSIER_MODELES, "yolov4-tiny.cfg")
FICHIER_POIDS = os.path.join(DOSSIER_MODELES, "yolov4-tiny.weights")
FICHIER_CLASSES = os.path.join(DOSSIER_MODELES, "coco.names")   # 80 classes COCO

# Taille de l'image envoyée au réseau. C'est LE paramètre qui pèse sur la
# cadence : le temps de calcul est à peu près proportionnel à sa surface.
#   320 = rapide (défaut ici : le modèle tourne deux fois par image),
#   416 = plus précis sur les petits objets, environ 1,7× plus lent.
# Doit être un multiple de 32.
TAILLE_ENTREE = 320

SEUIL_CONFIANCE = 0.5     # confiance minimale pour garder une détection
SEUIL_NMS = 0.4           # recouvrement toléré entre deux boîtes (NMS)

# Le modèle n'est pas obligé de tourner sur CHAQUE image capturée : on peut
# réafficher les dernières boîtes pendant quelques images. 1 = à chaque image.
DETECTER_TOUTES_LES_N_IMAGES = 2

# ---------------------------------------------------------------------------
#  AFFICHAGE
# ---------------------------------------------------------------------------
NOM_FENETRE = "IA03 - Filtres et robustesse (q : quitter)"
NB_IMAGES_LISSAGE_FPS = 30      # la cadence affichée est moyennée sur N images
DOSSIER_CAPTURES = os.path.join(DOSSIER_PROJET, "captures")   # touche « s »
