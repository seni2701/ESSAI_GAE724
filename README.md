# # Prédiction des rendements du maïs en Montérégie par IA et télédétection

## Description

Ce projet développe une approche intégrée combinant la télédétection et l'intelligence artificielle pour estimer le rendement du maïs dans la région de la Montérégie au Québec. L'objectif principal est de concevoir des modèles capables de reproduire la variabilité spatiale et temporelle du rendement à partir de données multi-sources.

## Contexte

La Montérégie représente près de 40% des superficies de maïs-grain de la province de Québec. Dans un contexte de changements climatiques et de pression sur les ressources, la capacité à anticiper le rendement agricole est cruciale pour la sécurité alimentaire et la gestion durable des ressources.

## Objectifs

- Extraire et analyser les indices spectraux (NDVI, EVI, LAI) pour caractériser la dynamique spatio-temporelle du maïs
- Entraîner et comparer différents modèles d'apprentissage automatique : Random Forest, XGBoost, SVM et TabResNet
- Évaluer la performance des modèles via des indicateurs statistiques (R², RMSE, MAE) et produire des cartes de rendement estimées à l'échelle parcellaire et régionale

## Méthodologie

### Données mobilisées

- **Données satellitaires** : Sentinel-2, Landsat 5/7/8 (via Google Earth Engine)
- **Données climatiques** : Stations météorologiques d'Environnement Canada (2010-2023)
- **Données pédologiques** : Banque de données pédologiques du Québec (BDPQ)
- **Observations terrain** : 10 parcelles expérimentales (F1-F10) avec mesures de rendement, biomasse et humidité du sol
- **Données historiques** : Statistiques de rendement FADQ (2010-2023)

### Indices spectraux calculés

- **NDVI** : (NIR - Red) / (NIR + Red) pour la Vigueur végétale
- **EVI** : 2.5 × (NIR - Red) / (NIR + 6 × Red - 7.5 × Blue + 1) pour la biomasse en zones denses
- **NDWI** : (NIR - SWIR) / (NIR + SWIR) pour la stress hydrique 
- **LAI** : Calculé à partir de NIR et Red pour la surface foliaire 

### Modèles d'apprentissage automatique

Quatre algorithmes ont été développés et comparés :

1. **Random Forest** : Robuste, R² = 0.578, RMSE = 0.884 t/ha
2. **XGBoost** : Meilleure performance, R² = 0.779, RMSE = 0.640 t/ha
3. **Support Vector Machine (SVM)** : Performances limitées sur ce jeu de données
4. **TabResNet** : Réseau neuronal, nécessite davantage de données pour converger

### Validation

- **Division temporelle** : entraînement 2010-2020, test 2021-2023
- **Validation croisée** à 5 plis (k-fold)
- **Métriques d'évaluation** : R², RMSE, MAE
- **Analyse SHAP** pour l'interprétabilité des modèles

## Résultats principaux

Le modèle **XGBoost** s'est révélé le plus performant avec :
- R² = 0.779
- RMSE = 0.640 t/ha
- MAE = 0.415 t/ha
- Erreur relative d'environ 5-6%

Les variables les plus influentes identifiées :
1. Précipitations totales (ppt_mm)
2. Température maximale (tmax)
3. Indice foliaire (LAI)
4. Température moyenne (tmean)
5. Indices spectraux (NDVI, EVI, NDWI)

## Structure du dépôt

```
essai/
│
├── data/                     # Données d'entrée
│   ├── climate/             # Données climatiques
│   ├── pedology/            # Données pédologiques
│   └── satellite/           # Indices spectraux
│
├── scripts/                  # Scripts de modélisation
│   ├── yield_rf.py          # Modèle Random Forest
│   ├── yield_svm.py         # Modèle SVM
│   └── yield_xgboost.py     # Modèle XGBoost
│
├── notebooks/                # Notebooks d'analyse
│   ├── preprocessing.ipynb  # Prétraitement des données
│   └── visualization.ipynb  # Visualisations
│
├── results/                  # Résultats et cartes
│   ├── maps/                # Cartes de rendement
│   └── metrics/             # Métriques de performance
│
└── docs/                     # Documentation
    └── essai_complet.pdf    # Document de thèse complet
```

## Installation et utilisation

### Prérequis

- Python 3.11
- Ubuntu 24.04 (ou système Linux équivalent)
- Accès à Google Earth Engine

### Bibliothèques principales

```bash
pip install scikit-learn xgboost numpy pandas matplotlib seaborn pytorch shap geopandas
```

### Environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Exécution des modèles

```bash
# Modèle Random Forest
python scripts/yield_rf.py

# Modèle XGBoost
python scripts/yield_xgboost.py

# Modèle SVM
python scripts/yield_svm.py
```

## Outils et logiciels

- **Traitement des données** : Python, Google Earth Engine, QGIS
- **Modélisation** : scikit-learn, XGBoost, PyTorch
- **Visualisation** : matplotlib, seaborn, ArcGIS Pro
- **Environnement** : VS Code, Google Colab

## Variables d'entrée des modèles

Les modèles intègrent :
- Indices de végétation (NDVI, EVI, LAI)
- Variables climatiques (températures min/max/moy, précipitations, neige)
- Caractéristiques pédologiques (drainage)
- Variables spatiales (région, zone, station météo)

## Limites et recommandations

### Limites identifiées
- Nombre limité d'observations (224)
- Couverture nuageuse affectant les séries temporelles
- Absence de certaines variables agronomiques (fertilisation, densité de semis)

### Recommandations futures
- Enrichir le jeu de données avec davantage de parcelles
- Intégrer des séries temporelles Sentinel-2 continues
- Explorer des modèles hybrides (XGBoost-NN, CNN-LSTM)
- Développer une plateforme interactive de visualisation
- Intégrer l'imagerie drone pour améliorer la résolution spatiale

## Applications potentielles

- Agriculture de précision
- Prévision des récoltes à l'échelle régionale
- Gestion du risque climatique
- Aide à la décision pour les producteurs agricoles
- Politiques de sécurité alimentaire

## Auteur
Ibrahima SENE 
Maître ès Sciences (M. Sc.) en géomatique appliquée et télédétection  
Département de géomatique appliquée  
Université de Sherbrooke  
Septembre 2025

Supervision : Mickaël Germain  
Co-supervision : Ramata Magagi

## Licence

Ce projet a été réalisé dans le cadre d'un essai de maîtrise à l'Université de Sherbrooke.

## Citation

Si vous utilisez ce travail, veuillez citer :

SENE, I. (2025). Prédiction des rendements du maïs en Montérégie par l'intelligence 
artificielle et la télédétection. Essai de maîtrise, Université de Sherbrooke, 
Département de géomatique appliquée.

## Remerciements

Un grand merci à :
- La Financière agricole du Québec (FADQ) pour les données de rendement
- Environnement et Changement climatique Canada (ECCC) pour les données météorologiques
- L'Institut de recherche et de développement en agroenvironnement (IRDA)
- Le personnel du Département de géomatique appliquée de l'Université de Sherbrooke
