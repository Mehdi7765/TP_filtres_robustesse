# Observations : à quelle intensité le modèle cesse-t-il de voir ?

Ce document rassemble les mesures faites avec `etalonnage.py` et les explications
demandées à l'oral. Les valeurs dépendent de l'objet, de l'éclairage et de la
distance : **refaites la mesure devant votre caméra** avec `python3 etalonnage.py`
(ou à la main dans `tp_filtres.py`) et complétez le tableau de la section 3.

Modèle : YOLOv4-tiny (COCO, 80 classes), entrée 320 px, seuil de confiance 0,5.

## 1. Mesures sur deux images de test (640 px de large)

Image A : deux personnes en gros plan (2 objets détectés à l'origine).
Image B : scène de rue avec trois personnes (3 objets détectés à l'origine).

| Filtre | Plage réglable | Premier objet perdu (A / B) | Plus aucun objet (A / B) |
|---|---|---|---|
| niveaux de gris | 0 → 100 % | 90 % / jamais | jamais / jamais |
| flou | 1 → 61 px | 5 px / 31 px | ~35 px / 53 px |
| contours | seuil 10 → 400 | dès 10 / dès 10 | dès 10 / dès 10 |
| seuillage | seuil 0 → 255 | partout instable | 170 / 180 |
| bruit | sigma 0 → 120 | sigma 15 / sigma 15 | instable au-delà de 30 |
| pixellisation | 1 → 64 px | 3 px / 5 px | 4 px / 7 px |
| posterisation | 0 → 7 bits supprimés | 4 bits / 7 bits | jamais / jamais |
| assombrissement | 0 → 100 % | 30 % / 85 % | 100 % / 100 % |
| inversion | 0 → 100 % | 30 % / 30 % | 50 % / jamais |

Deux précautions de lecture :

- **Un objet « détecté » n'est pas forcément le bon.** Sur une image très floue
  (35 à 61 px) ou très bruitée, le modèle peut renvoyer une boîte « person »
  au mauvais endroit : le compteur remonte alors que l'image est illisible.
  À l'oral, regardez la classe et la position de la boîte, pas seulement le compteur.
- **Le bruit est tiré à chaque image** : le compteur clignote autour du seuil.
  C'est pour cela que `etalonnage.py` moyenne plusieurs tirages.

## 2. Pourquoi tel filtre gêne-t-il plus que tel autre ?

Le fil conducteur : **le modèle ne voit pas comme nous**. Il a appris des
*statistiques* de pixels sur des photos naturelles (COCO). Tout ce qui garde
ces statistiques passe ; tout ce qui les change le rend aveugle, même quand
l'image reste évidente pour un œil humain.

### Ce que le modèle supporte très bien

- **Niveaux de gris** : quasiment aucun effet, même à 100 %. La couleur est un
  indice mineur : ce sont les formes, les contours et les textures qui
  déclenchent les filtres de convolution. De plus, l'entraînement de YOLO utilise
  de l'*augmentation de données* (variations de teinte et de saturation), le
  réseau a donc déjà vu des images presque incolores.
- **Postérisation** : réduire à 16 ou 8 niveaux par canal ne change pas la
  structure de l'image (les gros aplats restent, les contours restent). Le
  modèle décroche seulement quand il ne reste que 2 niveaux, c'est-à-dire
  quand la postérisation devient un seuillage.
- **Assombrissement** jusqu'à 80-90 % : le réseau normalise les pixels dans
  [0, 1] mais aucune normalisation par image n'est faite, donc les activations
  baissent. Elles restent pourtant *proportionnelles* : les contrastes relatifs
  sont conservés jusqu'à ce que la quantification sur 8 bits (des valeurs
  entières) écrase tout vers 0.

### Ce qui le gêne progressivement

- **Flou** : un flou gaussien supprime les hautes fréquences, c'est-à-dire les
  textures fines et les contours nets. Or les premières couches du réseau sont
  précisément des détecteurs de bords et de textures. Quand le noyau dépasse la
  taille des détails distinctifs de l'objet (yeux, logo, bords du téléphone),
  il ne reste que la silhouette et les scores tombent sous 0,5. Les gros objets
  proches résistent plus longtemps que les petits, ce qui explique l'écart entre
  les images A et B. L'œil, lui, reconnaît une silhouette floue sans effort.
- **Pixellisation** : c'est un flou brutal *plus* des faux contours (les bords
  des gros pixels). À 8 px, une image 640×480 ne contient plus que 80×60
  informations ; envoyée au réseau en 320×320, chaque objet fait quelques
  « gros pixels » seulement. Le modèle perd très vite, alors qu'un humain lit
  encore la scène à 16 px.

### Ce qui le rend aveugle alors que l'image reste lisible

- **Bruit** (le plus surprenant) : à sigma 15-20, le bruit est à peine visible
  à l'œil, et pourtant le modèle perd des objets. Un bruit indépendant sur
  chaque pixel est un signal de très haute fréquence : il excite fortement les
  détecteurs de bords des premières couches, dans toutes les directions et
  partout dans l'image. L'erreur se propage et s'amplifie de couche en couche.
  Notre système visuel, lui, moyenne naturellement le bruit (intégration
  spatiale) et ne le remarque pas. C'est le même mécanisme que les *exemples
  adverses* : une perturbation invisible qui bascule la décision.
- **Contours (Canny)** : aveugle dès la plus petite intensité. L'image devient
  binaire : fond noir, traits blancs, plus aucune texture ni dégradé. Les
  statistiques d'entrée n'ont plus rien à voir avec une photo ; les couches de
  *batch normalization* ont été calibrées sur des photos naturelles et
  produisent des activations aberrantes. Pour nous, un dessin au trait d'un
  visage ou d'une bouteille est immédiatement reconnaissable.
- **Seuillage** : même raison (image binaire), avec en plus une perte
  d'information réelle : tout ce qui est plus sombre ou plus clair que le seuil
  fusionne en un aplat. Le compteur remonte parfois vers 20-60 : ce sont
  souvent de fausses détections sur des taches noires et blanches.
- **Inversion** : à 100 % l'image est parfaitement lisible pour nous (un
  négatif), mais un ciel noir, une peau bleu-vert et une ombre blanche ne
  ressemblent à rien de ce que le réseau a appris. Le passage par 50 % (image
  grise uniforme, contraste nul) est le point le plus destructeur.

### Résumé pour l'oral

| Le filtre... | ...conserve | ...détruit | Effet sur le modèle |
|---|---|---|---|
| gris, postérisation | formes, contours, textures | couleur, nuances | quasi nul |
| assombrissement | contrastes relatifs | amplitude | faible jusqu'à ~85 % |
| flou, pixellisation | silhouette | détails fins (hautes fréquences) | progressif, dépend de la taille de l'objet |
| bruit | tout, pour l'œil | rien de visible, mais ajoute des hautes fréquences partout | fort et précoce |
| contours, seuillage, inversion | lisibilité pour l'œil | les statistiques d'une photo naturelle | aveugle presque immédiatement |

L'idée à retenir : **la robustesse du modèle ne suit pas la lisibilité humaine.**
Il est robuste aux transformations présentes dans ses données d'entraînement
(couleur, luminosité, échelle) et fragile à tout ce qui déplace l'image hors de
la distribution des photos naturelles (bruit, binarisation, négatif).

## 3. Mesures devant la caméra (à compléter en séance)

| Objet | Filtre | Intensité où la détection lâche |
|---|---|---|
| | | |
