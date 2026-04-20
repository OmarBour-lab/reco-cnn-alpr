# ALPR — Reconnaissance automatique de plaques d'immatriculation

Projet académique de Deep Learning : système de reconnaissance automatique de plaques d'immatriculation à deux étapes, basé sur un détecteur CNN pour localiser la plaque puis un OCR appliqué sur la plaque extraite.

Le notebook principal du projet est :

- `alpr_notebook_ameliore.ipynb`

## Objectif

Le projet vise à :

1. détecter une plaque d'immatriculation dans une image automobile ;
2. extraire la plaque à partir de la boîte détectée ;
3. lire automatiquement le texte de la plaque ;
4. évaluer les performances sur un sous-ensemble réel annoté manuellement.

## Pipeline du projet

Le pipeline final est le suivant :

```text
Image voiture
   -> Détection plaque : YOLOv8-xs (keras-cv)
   -> Bounding box normalisée
   -> Extraction / crop de la plaque
   -> Génération de plusieurs variantes de prétraitement
      (equalizeHist, CLAHE, blur, sharpen, Otsu, adaptive threshold, deskew, rotations légères)
   -> OCR sur plaque entière (EasyOCR)
   -> Sélection de la lecture la plus plausible
   -> Comparaison avec la vérité terrain (ground_truth_filled.csv)
```

## Technologies utilisées

- Python 3.11
- TensorFlow 2.21
- KerasCV 0.9
- OpenCV
- EasyOCR
- PyTorch
- Pandas
- Matplotlib
- Pillow
- scikit-learn

## Dataset

Le projet utilise :

- un dataset d'images de voitures annotées en Pascal VOC pour la détection de plaques ;
- un sous-ensemble de plaques extraites et annotées manuellement pour l'évaluation réelle de la lecture.

Structure attendue :

```text
reco-cnn/
├── data/
│   └── raw/
│       ├── images/
│       └── annotations/
├── extracted_plates/
│   ├── plate_crops/
│   └── ground_truth_filled.csv
├── models/
├── reports/
└── alpr_notebook_ameliore.ipynb
```

## Installation de l'environnement

Le projet a été préparé avec `uv`.

### 1. Installer uv

```bash
pip install --user uv
```

### 2. Installer Python 3.11

```bash
python -m uv python install 3.11
```

### 3. Créer l'environnement virtuel

```bash
python -m uv venv --python 3.11 .venv
```

### 4. Activer l'environnement

Sous Windows :

```bash
.venv\Scripts\activate
```

### 5. Installer les dépendances

```bash
python -m uv pip install tensorflow keras-cv opencv-python matplotlib scikit-learn pandas pillow lxml easyocr torch torchvision jupyter ipykernel
```

### 6. Ajouter le kernel Jupyter

```bash
python -m ipykernel install --user --name alpr-cnn --display-name "Python (alpr-cnn)"
```

## Instructions d'exécution

### Étape 1 — Placer le projet dans le bon dossier

Ouvrir le notebook depuis le dossier racine du projet `reco-cnn`, afin que :

```python
PROJECT_DIR = Path.cwd()
```

pointe correctement vers le projet.

### Étape 2 — Vérifier les données

Les dossiers suivants doivent exister :

```text
data/raw/images/
data/raw/annotations/
```

Les annotations XML doivent correspondre aux images.

### Étape 3 — Vérifier le fichier d'évaluation réelle

Le fichier suivant doit être présent si vous voulez calculer les métriques réelles de lecture :

```text
extracted_plates/ground_truth_filled.csv
```

Ce fichier doit contenir au minimum les colonnes :

- `id`
- `crop_path_local`
- `ground_truth_text`
- `use_for_eval`

### Étape 4 — Lancer Jupyter

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

### Étape 5 — Ouvrir le notebook final

Ouvrir :

```text
alpr_notebook_ameliore.ipynb
```

### Étape 6 — Exécuter le notebook

Dans Jupyter :

- sélectionner le kernel `Python (alpr-cnn)`
- exécuter `Restart & Run All`

## Comportement du notebook

Le notebook :

1. charge les données ;
2. prépare les splits train / validation / test ;
3. construit le détecteur YOLOv8-xs ;
4. charge les poids du détecteur s'ils existent ;
5. effectue un sanity check du détecteur ;
6. réentraîne automatiquement le détecteur si les poids chargés semblent incohérents ;
7. extrait les plaques annotées dans `extracted_plates/plate_crops/` ;
8. applique l'OCR sur plaque entière avec plusieurs prétraitements ;
9. génère les prédictions dans `extracted_plates/plate_predictions.csv` ;
10. compare les prédictions au fichier `ground_truth_filled.csv` ;
11. sauvegarde les résultats dans `reports/`.

## Fichiers générés

Pendant l'exécution, le projet peut générer :

```text
models/detector_final.weights.h5
reports/detector_history.csv
reports/real_eval_details.csv
reports/real_plate_eval.json
reports/final_results.json
extracted_plates/plates_manifest.csv
extracted_plates/plate_predictions.csv
extracted_plates/plate_crops/
```

## Résultats à observer

Les principales métriques affichées sont :

### Détection
- Mean IoU
- Median IoU
- IoU > 0.5
- IoU > 0.7
- nombre d'images sans détection

### Lecture réelle des plaques
- `exact_match_rate`
- `mean_char_accuracy`
- `empty_prediction_rate`

## Fichiers importants à conserver

À conserver pour le rendu final :

- `alpr_notebook_ameliore.ipynb`
- `README.md`
- `models/detector_final.weights.h5`
- `reports/final_results.json`
- `reports/real_plate_eval.json`
- `reports/real_eval_details.csv`
- `extracted_plates/ground_truth_filled.csv`

## Fichiers pouvant être supprimés

Vous pouvez supprimer les anciens fichiers intermédiaires ou de test si vous ne les utilisez plus, par exemple :

- anciens notebooks d'essai non utilisés ;
- anciens templates CSV non utilisés ;
- fichiers de debug temporaires ;
- anciennes versions de README ;
- anciens manifests ou prédictions obsolètes.

À ne pas supprimer :

- `data/raw/images/`
- `data/raw/annotations/`
- le notebook final
- les poids du détecteur
- le CSV de vérité terrain final

## Remarques

- la détection est réalisée avec YOLOv8-xs via KerasCV ;
- la lecture finale ne repose plus sur une segmentation caractère par caractère ;
- l'OCR est appliqué directement sur la plaque entière avec plusieurs variantes de prétraitement ;
- l'évaluation finale repose sur un sous-ensemble réel annoté manuellement.

## Limites

- les résultats restent dépendants de la qualité des crops de plaque ;
- certaines plaques très petites, floues ou inclinées restent difficiles à lire ;
- l'OCR peut produire des erreurs partielles sur certains formats de plaques ;
- les performances finales dépendent directement de la qualité du fichier `ground_truth_filled.csv`.

## Pistes d'amélioration

- utiliser un OCR spécialisé plaques encore plus robuste ;
- augmenter le nombre de plaques annotées manuellement ;
- enrichir le jeu de données avec plus de variabilité ;
- tester un détecteur plus grand si le GPU le permet ;
- ajouter une normalisation plus spécifique selon le format des plaques.
