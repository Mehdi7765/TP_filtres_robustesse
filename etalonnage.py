#!/usr/bin/env python3
"""
ÉTALONNAGE AUTOMATIQUE : À QUELLE INTENSITÉ LE MODÈLE DÉCROCHE-T-IL ?
=====================================================================
Fait ce que l'on fait à la main dans tp_filtres.py, mais pour toutes les
intensités de tous les filtres d'un coup, sur UNE image :

    python3 etalonnage.py                    # une image prise par la webcam
    python3 etalonnage.py --source photo.jpg # une image fixe
    python3 etalonnage.py --planche          # + enregistre une planche d'images

Pour chaque filtre, le script balaie l'intensité du minimum au maximum et note :
  - l'intensité à partir de laquelle le modèle perd un premier objet ;
  - l'intensité à partir de laquelle il ne voit plus RIEN.
Le résultat est un tableau Markdown, à coller dans docs/observations.md.
Les filtres aléatoires (bruit) sont moyennés sur plusieurs tirages.
"""
import argparse
import os
import time

import cv2
import numpy as np

import config
from detecteur import DetecteurYOLO
from filtres import creer_filtres


def prendre_image(args):
    if args.source:
        image = cv2.imread(args.source)
        if image is None:
            raise SystemExit(f"Impossible de lire {args.source}")
        # même largeur que la webcam, pour des mesures comparables
        h, l = image.shape[:2]
        if l != args.largeur:
            image = cv2.resize(image, (args.largeur, int(h * args.largeur / l)))
        return image
    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.largeur)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HAUTEUR_CAPTURE)
    if not capture.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra.")
    print("Caméra ouverte, capture dans 2 s : présentez l'objet...")
    fin = time.time() + 2.0
    image = None
    while time.time() < fin:                 # on laisse l'exposition se régler
        ok, image = capture.read()
    capture.release()
    if image is None:
        raise SystemExit("Aucune image lue.")
    return image


def compter(detecteur, image, tirages):
    """Nombre moyen d'objets détectés (plusieurs tirages pour les filtres aléatoires)."""
    return float(np.mean([len(detecteur.detecter(image)) for _ in range(tirages)]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="image fixe (sinon : webcam)")
    p.add_argument("--camera", type=int, default=config.INDEX_CAMERA)
    p.add_argument("--largeur", type=int, default=config.LARGEUR_CAPTURE)
    p.add_argument("--taille-entree", type=int, default=config.TAILLE_ENTREE)
    p.add_argument("--seuil", type=float, default=config.SEUIL_CONFIANCE)
    p.add_argument("--tirages", type=int, default=5, help="répétitions pour le filtre bruit")
    p.add_argument("--planche", action="store_true",
                   help="enregistre dans captures/ une planche par filtre (4 intensités)")
    args = p.parse_args()

    detecteur = DetecteurYOLO(config.FICHIER_CFG, config.FICHIER_POIDS, config.FICHIER_CLASSES,
                              taille_entree=args.taille_entree, seuil_confiance=args.seuil,
                              seuil_nms=config.SEUIL_NMS)
    image = prendre_image(args)
    reference = detecteur.detecter(image)
    nb_ref = len(reference)
    print(f"\nImage {image.shape[1]}x{image.shape[0]}, entrée réseau {args.taille_entree}px, "
          f"seuil {args.seuil}")
    print("Objets détectés sur l'original :",
          ", ".join(f'{d["classe"]} ({d["confiance"]:.0%})' for d in reference) or "AUCUN")
    if nb_ref == 0:
        raise SystemExit("Rien à mesurer : placez un objet reconnu par le modèle devant la caméra.")

    lignes = []
    for filtre in creer_filtres():
        tirages = args.tirages if filtre.nom == "bruit" else 1
        premiere_perte, aveugle = None, None
        courbe = []
        for valeur in filtre.valeurs_possibles():
            filtre.intensite = valeur
            nb = compter(detecteur, filtre.appliquer(image), tirages)
            courbe.append((valeur, nb))
            if premiere_perte is None and nb < nb_ref - 0.5:
                premiere_perte = valeur
            if aveugle is None and nb < 0.5:
                aveugle = valeur
            elif aveugle is not None and nb >= 0.5:
                aveugle = None            # le modèle revoit : on cherche la perte durable
        texte_courbe = "  ".join(f"{v}:{n:.0f}" if n == int(n) else f"{v}:{n:.1f}" for v, n in courbe)
        print(f"\n[{filtre.nom}] {texte_courbe}")
        lignes.append((filtre.nom, filtre,
                       premiere_perte, aveugle, courbe))

        if args.planche:
            valeurs = filtre.valeurs_possibles()
            choix = [valeurs[int(i * (len(valeurs) - 1) / 3)] for i in range(4)]
            vignettes = []
            for valeur in choix:
                filtre.intensite = valeur
                img = filtre.appliquer(image)
                dets = detecteur.detecter(img)
                detecteur.annoter(img, dets)
                cv2.putText(img, f"{filtre.nom} {filtre.texte_intensite()} -> {len(dets)} obj.",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
                vignettes.append(img)
            os.makedirs(config.DOSSIER_CAPTURES, exist_ok=True)
            chemin = os.path.join(config.DOSSIER_CAPTURES, f"planche_{filtre.nom.replace(' ', '_')}.jpg")
            cv2.imwrite(chemin, np.hstack(vignettes), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("   planche :", chemin)
        filtre.reinitialiser()

    def fmt(filtre, valeur):
        if valeur is None:
            return "jamais (dans les bornes)"
        filtre.intensite = valeur
        return filtre.texte_intensite()

    print(f"\n\n| Filtre | Plage | Premier objet perdu | Plus aucun objet ({nb_ref} au départ) |")
    print("|---|---|---|---|")
    for nom, filtre, perte, aveugle, _ in lignes:
        filtre.intensite = filtre.minimum
        bas = filtre.texte_intensite()
        filtre.intensite = filtre.maximum
        haut = filtre.texte_intensite()
        print(f"| {nom} | {bas} → {haut} | {fmt(filtre, perte)} | {fmt(filtre, aveugle)} |")


if __name__ == "__main__":
    main()
