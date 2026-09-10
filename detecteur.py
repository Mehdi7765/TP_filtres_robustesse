"""
DÉTECTEUR D'OBJETS YOLOv4-tiny (module DNN d'OpenCV)
====================================================
Le modèle est chargé UNE SEULE FOIS à la construction (un chargement prend
environ 0,5 s : le refaire à chaque image diviserait la cadence par dix).

  - `detecter(image)`            -> liste des objets trouvés (classe, confiance, boîte)
  - `annoter(image, detections)` -> dessine les boîtes et étiquettes sur l'image

Aucune dépendance en dehors d'OpenCV et NumPy : pas de PyTorch à installer.
"""
import cv2
import numpy as np


class DetecteurYOLO:

    def __init__(self, chemin_cfg, chemin_poids, chemin_classes,
                 taille_entree=320, seuil_confiance=0.5, seuil_nms=0.4):
        with open(chemin_classes, encoding="utf-8") as f:
            self.classes = [ligne.strip() for ligne in f if ligne.strip()]

        self._reseau = cv2.dnn.readNetFromDarknet(chemin_cfg, chemin_poids)
        self._reseau.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._reseau.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._couches_sortie = self._reseau.getUnconnectedOutLayersNames()

        self.taille_entree = taille_entree
        self.seuil_confiance = seuil_confiance
        self.seuil_nms = seuil_nms

        # Une couleur fixe par classe : les boîtes ne changent pas de couleur
        # d'une image à l'autre.
        generateur = np.random.default_rng(42)
        self._couleurs = generateur.integers(80, 255, size=(len(self.classes), 3))

    # ------------------------------------------------------------------ #
    def detecter(self, image):
        """Renvoie une liste de dict : {"classe", "confiance", "boite": (x, y, w, h)}.

        L'image doit être en couleur (3 canaux BGR) : c'est ce que le réseau
        attend. Les filtres renvoient donc toujours une image à 3 canaux, même
        les filtres « noir et blanc ».
        """
        hauteur, largeur = image.shape[:2]
        taille = int(self.taille_entree)
        seuil = float(self.seuil_confiance)

        # 1) Préparation : pixels ramenés dans [0,1], redimensionnement à la
        #    taille d'entrée du réseau, passage BGR (OpenCV) -> RGB (YOLO).
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (taille, taille),
                                     swapRB=True, crop=False)

        # 2) Passage dans le réseau
        self._reseau.setInput(blob)
        sorties = self._reseau.forward(self._couches_sortie)

        # 3) Décodage : chaque ligne = [cx, cy, w, h, objectness, score_classe_0..79]
        boites, confiances, indices_classes = [], [], []
        for sortie in sorties:
            for detection in sortie:
                scores = detection[5:]
                indice = int(np.argmax(scores))
                confiance = float(scores[indice])
                if confiance < seuil:
                    continue
                # Coordonnées normalisées (0-1) -> pixels, centre -> coin haut-gauche
                cx, cy = detection[0] * largeur, detection[1] * hauteur
                w, h = detection[2] * largeur, detection[3] * hauteur
                boites.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                confiances.append(confiance)
                indices_classes.append(indice)

        # 4) NMS : un même objet est souvent détecté plusieurs fois,
        #    on ne garde que la meilleure boîte pour chacun.
        retenus = cv2.dnn.NMSBoxes(boites, confiances, seuil, self.seuil_nms)
        resultats = []
        for i in np.array(retenus).flatten():
            resultats.append({
                "classe": self.classes[indices_classes[i]],
                "confiance": confiances[i],
                "boite": tuple(boites[i]),
                "_indice": indices_classes[i],
            })
        return resultats

    # ------------------------------------------------------------------ #
    def annoter(self, image, detections):
        """Dessine les boîtes et étiquettes sur l'image (modifiée en place)."""
        for d in detections:
            x, y, w, h = d["boite"]
            couleur = tuple(int(c) for c in self._couleurs[d["_indice"]])
            etiquette = f'{d["classe"]} {d["confiance"]:.0%}'
            cv2.rectangle(image, (x, y), (x + w, y + h), couleur, 2)
            # Fond plein derrière le texte pour rester lisible sur toute image
            (tw, th), _ = cv2.getTextSize(etiquette, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            y_texte = max(y, th + 8)
            cv2.rectangle(image, (x, y_texte - th - 8), (x + tw + 6, y_texte), couleur, -1)
            cv2.putText(image, etiquette, (x + 3, y_texte - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return image
