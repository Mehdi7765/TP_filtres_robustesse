# Filtres d'image et robustesse d'un modèle de détection — IA03

*Mehdi Ez-Zouak, Paul Louis Ledoux, Clément Menaucourt — UTT, année 2, septembre 2026.*

Un outil pour observer **en direct** à partir de quel moment un modèle de détection d'objets
(YOLOv4-tiny) cesse de voir une image dégradée. La fenêtre montre deux images issues de la
même capture webcam :

- à **gauche**, l'image d'origine avec les détections du modèle ;
- à **droite**, la même image après un filtre, avec les détections du modèle sur cette version.

On change de filtre et on règle son intensité **au clavier**, et on regarde les compteurs
(nom du filtre, intensité, objets détectés de chaque côté, cadence) jusqu'à ce que la
détection lâche.

```
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ ORIGINALE                    │ FILTRE 2/9 : FLOU   intensite : 15 px        │
│ objets detectes : 2          │ objets detectes : 1                          │
│                              │                                              │
│      [person 86%]            │      [person 54%]                            │
│                              │                                              │
│ 12.3 images/s  detection ON  │ 1-9 filtre  n/p  +/- intensite  r  d  s  q   │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 1. Installation

Prérequis : **Python 3.8 ou plus** et une webcam. Aucun PyTorch à installer : le modèle
tourne avec le module DNN d'OpenCV, sur CPU. Les poids (24 Mo) sont dans le dépôt.

```bash
git clone https://github.com/Mehdi7765/TP_filtres_robustesse.git
cd TP_filtres_robustesse
```

Windows / macOS :

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS
pip install -r requirements.txt
```

Linux (Debian / Ubuntu) : `sudo apt install python3-opencv python3-numpy`, ou le `venv`
ci-dessus si `pip` refuse avec *externally-managed-environment*.

Vérifier : `python -c "import cv2, numpy; print('OpenCV', cv2.__version__, '- OK')"`

> Sur Linux/macOS la commande s'appelle généralement `python3` ; sur Windows `python`.

## 2. Utilisation

```bash
python3 tp_filtres.py                       # webcam 0, détection activée
python3 tp_filtres.py --camera 1            # autre webcam
python3 tp_filtres.py --taille-entree 416   # modèle plus précis mais plus lent
python3 tp_filtres.py --toutes-les 1        # détecter sur chaque image (défaut : 1 sur 2)
python3 tp_filtres.py --source photo.jpg    # une image fixe à la place de la webcam
python3 tp_filtres.py --sans-detection      # le squelette seul : caméra + filtres
python3 tp_filtres.py --help
```

### Touches

| Touche | Action |
|---|---|
| `1` … `9` | choisir un filtre |
| `n` / `p` | filtre suivant / précédent |
| `+` / `-` | augmenter / diminuer l'intensité du filtre courant |
| `r` | remettre l'intensité par défaut |
| `d` | activer / désactiver la détection (pour mesurer son coût) |
| `s` | enregistrer une capture de la fenêtre dans `captures/` |
| `q` ou `Échap` | quitter en libérant la caméra |

### Les filtres et le sens de leur intensité

| # | Filtre | Intensité | Bornes |
|---|---|---|---|
| 1 | Niveaux de gris | % de désaturation (mélange couleur / gris) | 0 → 100 % |
| 2 | Flou gaussien | taille du noyau en px (toujours impaire) | 1 → 61 |
| 3 | Contours (Canny) | seuil haut de Canny (le bas vaut la moitié) | 10 → 400 |
| 4 | Seuillage | valeur qui sépare le noir du blanc | 0 → 255 |
| 5 | Bruit gaussien *(perso)* | écart-type du bruit ajouté | 0 → 120 |
| 6 | Pixellisation *(perso)* | taille d'un gros pixel | 1 → 64 px |
| 7 | Postérisation *(perso)* | bits supprimés par canal (7 = 2 niveaux) | 0 → 7 |
| 8 | Assombrissement *(perso)* | % de lumière retirée | 0 → 100 % |
| 9 | Inversion *(perso)* | % de mélange avec le négatif | 0 → 100 % |

Dans tous les cas, « augmenter » dégrade davantage l'image, et l'intensité est bornée :
on peut appuyer autant de fois qu'on veut, le programme ne plante pas. Le passage en
niveaux de gris n'a pas d'intensité naturelle ; il est rendu progressif par un mélange
entre l'image couleur et sa version grise (voir `filtres.py`).

## 3. Étalonnage automatique

`etalonnage.py` fait à la place de vous ce que l'on fait à la main : il prend une image
(webcam ou fichier), balaie toutes les intensités de chaque filtre et note celle où le
modèle perd un premier objet, puis celle où il ne voit plus rien. Il imprime un tableau
Markdown prêt à coller.

```bash
python3 etalonnage.py                     # capture webcam après 2 s
python3 etalonnage.py --source photo.jpg
python3 etalonnage.py --planche           # + une planche d'images par filtre dans captures/
```

Les résultats et surtout **les explications demandées à l'oral** sont dans
[`docs/observations.md`](docs/observations.md).

## 4. Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `tp_filtres.py` | programme principal : caméra, double affichage, clavier, compteurs, détection |
| `filtres.py` | les neuf filtres, chacun avec son intensité, ses bornes et son pas |
| `detecteur.py` | YOLOv4-tiny via OpenCV DNN : `detecter()` et `annoter()` |
| `etalonnage.py` | balayage automatique des intensités |
| `config.py` | caméra, modèle, taille d'entrée, cadence de détection |
| `modeles/` | `yolov4-tiny.cfg`, `yolov4-tiny.weights`, `coco.names` |
| `docs/observations.md` | mesures et explications pour l'oral (aussi en PDF : `docs/observations.pdf`) |
| `docs/notes_seance.txt` | relevé brut des valeurs mesurées en séance |
| `docs/TP_filtres_robustesse.pdf` | le sujet |

## 5. Choix techniques

- **Construction dans l'ordre du sujet** : le squelette (`--sans-detection`) tourne à la
  cadence de la webcam ; les filtres coûtent moins d'une milliseconde chacun sauf le bruit
  (~5 ms) ; la détection est ce qui ralentit.
- **Deux images côte à côte dans une seule fenêtre** avec `np.hstack`. Cela impose la même
  forme des deux côtés : chaque filtre renvoie donc **toujours une image BGR à 3 canaux**,
  même les filtres noir et blanc (`cv2.COLOR_GRAY2BGR`). C'est aussi ce qu'attend le réseau.
- **Clavier** : `cv2.waitKey(1) & 0xFF`, non bloquant, avec le masque qui retire les bits
  parasites de certains systèmes. Une touche inconnue affiche son code dans le terminal.
- **Cadence** : le modèle tourne deux fois par image, ce qui divise la cadence par deux ou
  trois. Trois leviers, tous réglables :
  - le modèle est chargé **une seule fois** au démarrage ;
  - l'image envoyée au réseau est réduite à **320 px** (`--taille-entree`, `config.py`) ;
    416 px est 1,7× plus lent ;
  - la détection ne tourne que sur **une image sur deux** (`--toutes-les`), les dernières
    boîtes sont réaffichées entre-temps.
  Sur un portable sans GPU : ~17 images/s sans détection, ~12 avec (320 px, 1 image sur 2),
  ~8 avec détection sur chaque image.
- **Cadence lissée** : écart de temps entre deux tours, moyenné sur les 30 dernières images
  (`collections.deque`), sinon la valeur clignote.
- **Sortie propre** : `capture.release()` et `cv2.destroyAllWindows()` dans un `finally`,
  donc aussi sur `Ctrl+C`.

## 6. Limites connues

- Le compteur compte des boîtes, pas des bonnes réponses : sur une image très dégradée le
  modèle peut renvoyer une fausse « person ». Regarder la classe et la position.
- YOLOv4-tiny à 320 px voit mal les petits objets ; pour un objet tenu loin de la caméra,
  passer à `--taille-entree 416`.
- Le bruit étant retiré à chaque image, le compteur clignote autour du seuil de décrochage :
  `etalonnage.py` moyenne cinq tirages.
