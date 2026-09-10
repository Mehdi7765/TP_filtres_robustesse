#!/usr/bin/env python3
"""
IA03 – FILTRES D'IMAGE ET ROBUSTESSE D'UN MODÈLE DE DÉTECTION
==============================================================
Affiche côte à côte l'image de la webcam et la même image après un filtre,
avec les détections YOLOv4-tiny des deux côtés. On change de filtre et on
règle son intensité au clavier, pour trouver le moment où le modèle décroche.

    python3 tp_filtres.py                      # webcam 0
    python3 tp_filtres.py --camera 1
    python3 tp_filtres.py --source photo.jpg   # une image fixe en boucle
    python3 tp_filtres.py --taille-entree 416  # plus précis, plus lent
    python3 tp_filtres.py --sans-detection     # juste les filtres (squelette)

Touches (dans la fenêtre) :
    1..9      choisir un filtre        n / p    filtre suivant / précédent
    + / -     intensité + / -          r        intensité par défaut
    d         détection on/off         s        enregistrer une capture
    q / Échap quitter
"""
import argparse
import os
import time
from collections import deque

import cv2
import numpy as np

import config
from detecteur import DetecteurYOLO
from filtres import creer_filtres

# ---------------------------------------------------------------------------
#  Aide à l'affichage
# ---------------------------------------------------------------------------
POLICE = cv2.FONT_HERSHEY_SIMPLEX
BLANC, NOIR, VERT, JAUNE, ROUGE = (255, 255, 255), (0, 0, 0), (80, 220, 80), (0, 220, 255), (0, 0, 255)


def bandeau(image, lignes, en_haut=True):
    """Écrit des lignes de texte sur un bandeau semi-transparent (haut ou bas)."""
    hauteur_ligne = 24
    h_bandeau = 10 + hauteur_ligne * len(lignes)
    h = image.shape[0]
    y0 = 0 if en_haut else h - h_bandeau
    zone = image[y0:y0 + h_bandeau]
    # fond noir mélangé à 60 % : le texte reste lisible sur toute image
    cv2.addWeighted(zone, 0.4, np.zeros_like(zone), 0.6, 0, zone)
    for i, (texte, couleur) in enumerate(lignes):
        cv2.putText(image, texte, (10, y0 + 22 + i * hauteur_ligne),
                    POLICE, 0.6, couleur, 1, cv2.LINE_AA)
    return image


def ouvrir_source(args):
    """Webcam (index), vidéo ou image fixe. Renvoie (capture, image_fixe)."""
    if args.source is not None and not str(args.source).isdigit():
        if os.path.splitext(args.source)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            image = cv2.imread(args.source)
            if image is None:
                raise SystemExit(f"Impossible de lire l'image {args.source}")
            return None, image
        capture = cv2.VideoCapture(args.source)
    else:
        index = int(args.source) if args.source is not None else args.camera
        capture = cv2.VideoCapture(index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.largeur)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.hauteur)
    if not capture.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra / la source vidéo.")
    return capture, None


def lire_arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=config.INDEX_CAMERA, help="index de la webcam")
    p.add_argument("--source", help="fichier image ou vidéo à la place de la webcam")
    p.add_argument("--largeur", type=int, default=config.LARGEUR_CAPTURE)
    p.add_argument("--hauteur", type=int, default=config.HAUTEUR_CAPTURE)
    p.add_argument("--taille-entree", type=int, default=config.TAILLE_ENTREE,
                   help="taille de l'image envoyée au réseau (multiple de 32)")
    p.add_argument("--toutes-les", type=int, default=config.DETECTER_TOUTES_LES_N_IMAGES,
                   help="ne détecter qu'une image sur N (1 = toutes)")
    p.add_argument("--seuil", type=float, default=config.SEUIL_CONFIANCE, help="confiance minimale")
    p.add_argument("--sans-detection", action="store_true", help="désactive le modèle au démarrage")
    p.add_argument("--max-images", type=int, default=0,
                   help="quitte après N images (tests et mesures de cadence ; 0 = illimité)")
    p.add_argument("--sans-affichage", action="store_true", help="pas de fenêtre (tests)")
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = lire_arguments()

    # 1) Le modèle, chargé UNE fois ---------------------------------------
    detecteur = None
    if not args.sans_detection:
        print("Chargement du modèle...", end=" ", flush=True)
        detecteur = DetecteurYOLO(config.FICHIER_CFG, config.FICHIER_POIDS, config.FICHIER_CLASSES,
                                  taille_entree=args.taille_entree,
                                  seuil_confiance=args.seuil, seuil_nms=config.SEUIL_NMS)
        print(f"OK ({len(detecteur.classes)} classes, entrée {args.taille_entree}px)")
    detection_active = detecteur is not None

    # 2) La caméra --------------------------------------------------------
    capture, image_fixe = ouvrir_source(args)
    if capture is not None:
        print(f"Caméra ouverte : {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(__doc__.split("Touches")[1])

    filtres = creer_filtres()
    index_filtre = 1                      # on démarre sur le flou
    detections_gauche, detections_droite = [], []
    duree_detection = 0.0
    durees_images = deque(maxlen=config.NB_IMAGES_LISSAGE_FPS)
    compteur = 0
    t_precedent = time.perf_counter()

    if not args.sans_affichage:
        cv2.namedWindow(config.NOM_FENETRE, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # --- lecture --------------------------------------------------
            if image_fixe is not None:
                image = image_fixe.copy()
            else:
                ok, image = capture.read()
                if not ok:
                    print("Flux terminé ou caméra perdue.")
                    break
            filtre = filtres[index_filtre]

            # --- filtre ---------------------------------------------------
            filtree = filtre.appliquer(image)

            # --- détection (des deux côtés) -------------------------------
            if detection_active and compteur % max(1, args.toutes_les) == 0:
                t0 = time.perf_counter()
                detections_gauche = detecteur.detecter(image)
                detections_droite = detecteur.detecter(filtree)
                duree_detection = (time.perf_counter() - t0) * 1000
            elif not detection_active:
                detections_gauche, detections_droite = [], []

            # --- cadence : écart avec le tour précédent, lissé ------------
            t_maintenant = time.perf_counter()
            durees_images.append(t_maintenant - t_precedent)
            t_precedent = t_maintenant
            fps = len(durees_images) / max(sum(durees_images), 1e-6)

            # --- annotation -----------------------------------------------
            gauche, droite = image.copy(), filtree
            if detecteur is not None:
                detecteur.annoter(gauche, detections_gauche)
                detecteur.annoter(droite, detections_droite)

            couleur_droite = VERT if len(detections_droite) >= len(detections_gauche) else ROUGE
            bandeau(gauche, [("ORIGINALE", BLANC),
                             (f"objets detectes : {len(detections_gauche)}", VERT)])
            bandeau(droite, [(f"FILTRE {index_filtre + 1}/{len(filtres)} : {filtre.nom.upper()}"
                              f"   intensite : {filtre.texte_intensite()}", JAUNE),
                             (f"objets detectes : {len(detections_droite)}", couleur_droite)])
            etat = (f"detection {'ON' if detection_active else 'OFF'}"
                    f"  ({duree_detection:.0f} ms pour 2 passages, 1 image sur {args.toutes_les})"
                    if detecteur is not None else "detection : modele non charge")
            bandeau(gauche, [(f"{fps:5.1f} images/s    {etat}", BLANC)], en_haut=False)
            bandeau(droite, [("1-9 filtre  n/p suivant  +/- intensite  r defaut  d detection  s capture  q quitter", BLANC)],
                    en_haut=False)

            # --- assemblage : même forme des deux côtés garantie ----------
            double = np.hstack((gauche, droite))

            if not args.sans_affichage:
                cv2.imshow(config.NOM_FENETRE, double)
                touche = cv2.waitKey(1) & 0xFF      # masque les bits parasites selon l'OS
            else:
                touche = 255

            # --- clavier --------------------------------------------------
            if touche in (ord("q"), 27):                        # q ou Échap
                break
            elif ord("1") <= touche <= ord("9"):
                index_filtre = min(touche - ord("1"), len(filtres) - 1)
            elif touche == ord("n"):
                index_filtre = (index_filtre + 1) % len(filtres)
            elif touche == ord("p"):
                index_filtre = (index_filtre - 1) % len(filtres)
            elif touche in (ord("+"), ord("=")):
                filtre.augmenter()
            elif touche in (ord("-"), ord("_")):
                filtre.diminuer()
            elif touche == ord("r"):
                filtre.reinitialiser()
            elif touche == ord("d") and detecteur is not None:
                detection_active = not detection_active
            elif touche == ord("s"):
                os.makedirs(config.DOSSIER_CAPTURES, exist_ok=True)
                nom = os.path.join(config.DOSSIER_CAPTURES,
                                   f"{time.strftime('%Y%m%d_%H%M%S')}_{filtre.nom.replace(' ', '_')}"
                                   f"_{filtre.intensite}.png")
                cv2.imwrite(nom, double)
                print("Capture enregistrée :", nom)
            elif touche != 255 and touche != 0:
                print(f"touche non utilisée : code {touche}")

            compteur += 1
            if args.max_images and compteur >= args.max_images:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # 3) Sortie propre : caméra libérée, fenêtre fermée -----------------
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        if durees_images:
            print(f"Fin. Cadence moyenne sur les {len(durees_images)} dernières images : "
                  f"{len(durees_images) / sum(durees_images):.1f} images/s")


if __name__ == "__main__":
    main()
