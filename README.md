# ALPR — Reconnaissance automatique de plaques d'immatriculation

Projet académique de Deep Learning pour détecter une plaque d'immatriculation dans une image, extraire la plaque, lire automatiquement son texte et évaluer les performances sur un sous-ensemble réel annoté manuellement.

Notebook principal :
- `alpr_notebook_ameliore.ipynb`

## Pipeline

```text
Image voiture
   -> Détection plaque : YOLOv8-xs (keras-cv)
   -> Bounding box
   -> Extraction / crop de la plaque
   -> Génération de plusieurs prétraitements
      (equalizeHist, CLAHE, blur, sharpen, Otsu, adaptive threshold, deskew, rotations légères)
   -> OCR sur plaque entière (EasyOCR)
   -> Sélection de la lecture la plus plausible
   -> Évaluation sur ground_truth_filled.csv
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

## Structure attendue

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

1. Ouvrir le projet depuis le dossier racine `reco-cnn`.
2. Vérifier la présence des dossiers :
   - `data/raw/images/`
   - `data/raw/annotations/`
3. Vérifier la présence du fichier :
   - `extracted_plates/ground_truth_filled.csv`
4. Lancer Jupyter :

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

5. Ouvrir `alpr_notebook_ameliore.ipynb`
6. Sélectionner le kernel `Python (alpr-cnn)`
7. Exécuter `Restart & Run All`

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
