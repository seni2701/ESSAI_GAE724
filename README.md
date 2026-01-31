# Prédiction des rendements du maïs en Montérégie par IA et télédétection


## Description

Ce projet développe une approche intégrée combinant la télédétection et l'intelligence artificielle pour estimer le rendement du maïs dans la région de la Montérégie au Québec sur la période 2010–2023. L'objectif principal est de concevoir des modèles capables de reproduire la variabilité spatiale et temporelle du rendement à partir de données multisources (satellitaires, climatiques, pédologiques et de rotation culturale).

## Contexte

La Montérégie représente près de 40 % des superficies de maïs-grain de la province du Québec, soit plus de 3,6 millions de tonnes produites sur environ 355 700 hectares. Dans un contexte de changements climatiques et de pression sur les ressources, la capacité à anticiper le rendement agricole est cruciale pour la sécurité alimentaire et la gestion durable des ressources. Les rendements moyens régionaux varient entre 9 et 11 t/ha, pouvant atteindre 12 t/ha dans les conditions les plus favorables.

## Objectifs

### Objectif général

Développer et comparer différents modèles d'IA afin de prédire le rendement du maïs en Montérégie à partir de données multisources combinant télédétection, variables climatiques, pédologiques et rotation culturale.

### Objectifs spécifiques

- Identifier les variables agroclimatiques, pédologiques et spectrales les plus déterminantes du rendement du maïs à partir des données satellitaires
- Comparer la performance des modèles d'apprentissage automatique et profond pour évaluer leur capacité à représenter la variabilité spatiale et temporelle des rendements
- Analyser l'effet de la rotation maïs–soya sur la stabilité et la précision des prédictions de rendement à l'échelle régionale

## Méthodologie

### Données mobilisées

Les données sont réparties en quatre catégories principales :

| Catégorie | Sources | Variables |
|-----------|---------|-----------|
| **Satellitaires** | Sentinel-2, Sentinel-1 (SAR), Landsat 5/7/8 via Google Earth Engine | NDVI, EVI, LAI, VV, VH, GLCM |
| **Climatiques** | ECCC, Agrométéo Québec, MELCCFP (2010–2023) | tmax, tmin, tmean, ppt_mm |
| **Pédologiques** | BDPQ / IRDA | Texture, drainage, matière organique, pente |
| **Rotation culturale** | AAFC (Annual Crop Inventory) via GEE | crop_type_lag1, is_corn, is_soy |

**Stations météorologiques utilisées** : Marieville, Montréal–Saint-Hubert, Sorel, Verchères.

**Parcelles étudiées** : 10 parcelles (F1–F10) réparties autour de Beloeil, Saint-Hyacinthe et Saint-Bernard-de-Michaudville, avec des séries de rendements continues sur 2010–2023 (FADQ). Les parcelles F1, F3, F5, F7 et F9 pratiquent une rotation régulière maïs–soya, tandis que F2, F4, F6, F8 et F10 présentent des séquences de monoculture plus fréquentes.

### Indices spectraux calculés

| Indice | Source | Utilité |
|--------|--------|---------|
| NDVI | Sentinel-2 (10 m) | Vigueur végétale et densité du couvert |
| EVI | Sentinel-2 (10 m) | Biomasse en zones de forte densité |
| LAI | Sentinel-2 | Surface foliaire active |
| VV / VH | Sentinel-1 (SAR) | Structure du couvert et humidité du sol |

Le masquage des nuages a été effectué via les bandes QA60, SCL et l'algorithme `s2cloudless`. La collection `COPERNICUS/S2_SR_HARMONIZED` garantit la cohérence radiométrique sur la période entière.

### Prétraitement des données

- Imputation des valeurs manquantes : `SimpleImputer` (moyenne)
- Normalisation des variables continues : `StandardScaler` et méthode min-max
- Encodage des variables catégorielles : `LabelEncoder`
- Masquage dynamique maïs/soya dans GEE avant extraction des indices

### Modèles testés et hyperparamètres

| Modèle | Algorithme | Paramètres clés |
|--------|------------|-----------------|
| **Random Forest** | Bagging (ensemble) | n_estimators=150, max_depth=10, max_samples=0.8, max_features=0.5 |
| **XGBoost** | Gradient boosting séquentiel | learning_rate=0.05, max_depth=5, subsample=0.75, colsample_bytree=0.75, gamma=0.2, L1 (α=0.3), L2 (λ=1.5), 300 itérations, early stopping=30 |
| **SVM** | Noyau radial (RBF) | C=10, ε=0.1, gamma=scale |
| **TabResNet** | Réseau neuronal résiduel | learning_rate=0.001, batch_size=32, 500 époqus, Dropout p=0.2, Batch Normalization |

### Validation

- **Division temporelle** : 70 % entraînement (2010–2020), 30 % test (2021–2023)
- **Validation croisée** : k-fold à 5 plis pour RF, XGB et TabResNet
- **Validation temporelle glissante** : TimeSeriesSplit + Leave-One-Year-Out pour SVM
- **Métriques** : R², RMSE, MAE
- **Interprétabilité** : Analyse SHAP pour identifier les variables déterminantes

## Résultats principaux

### Comparaison des performances

| Modèle | R² | RMSE (t/ha) | MAE (t/ha) | Erreur relative |
|--------|-----|-------------|------------|-----------------|
| **XGBoost** | **0.748** | **0.684** | **0.428** | **5.9 %** |
| Random Forest | 0.613 | 0.847 | 0.619 | 7.4 % |
| SVM | 0.193 | 0.912 | 0.721 | — |
| TabResNet | 0.022 | 1.374 | 1.174 | — |

> L'erreur relative est calculée par rapport au rendement moyen régional de la FADQ (≈ 11.5 t/ha). XGBoost améliore la précision d'environ 19 % (RMSE) et 31 % (MAE) par rapport au Random Forest.

### Variables les plus influentes (analyse SHAP – XGBoost)

1. **Précipitations cumulées** (`ppt_mm`) — facteur dominant
2. **Température maximale** (`tmax`)
3. **Entropie du NDVI** (`v6_v8_ndvi_entropy`) — hétérogénéité du couvert
4. **Variables radar Sentinel-1** (`VV`, `VH`) — humidité du sol
5. **Rotation culturale** (`crop_type_lag1`) — effet du précédent cultural
6. **Indices optiques** (`EVI`, `NDVI_max`, `LAI`) — vigueur végétative
7. **Températures** (`tmean`, `tmin`)

### Observations clés

- XGBoost reproduit fidèlement les années de stress climatique (2011, 2014, 2019) et les années à forte productivité (2016, 2017, 2022)
- Les parcelles en rotation maïs–soya (F1, F3, F5, F7, F9) présentent des résidus plus faibles que celles en monoculture
- La variabilité spatiale des rendements s'aligne avec les données FADQ : zones productives au centre/nord Montérégie (12–13 t/ha), moins productives au sud-ouest (<10 t/ha)
- Le drainage du sol est un facteur déterminant : rendement moyen de 9 t/ha sur sols mal drainés contre 12 t/ha sur sols bien drainés

## Structure du dépôt

```
ESSAI_GAE724/
│
├── data/                         # Données d'entrée
│   ├── climate/                 # Données climatiques (ECCC)
│   ├── pedology/                # Données pédologiques (BDPQ/IRDA)
│   ├── satellite/               # Indices spectraux (GEE)
│   └── rotation/                # Données AAFC rotation culturale
│
├── scripts/                      # Scripts de modélisation
│   ├── yield_rf.py              # Modèle Random Forest
│   ├── yield_svm.py             # Modèle SVM
│   ├── yield_xgboost.py         # Modèle XGBoost
│   └── modele_rendement_svm_FINAL.py
│
├── notebooks/                    # Notebooks d'analyse
│   ├── preprocessing.ipynb      # Prétraitement des données
│   └── visualization.ipynb      # Visualisations des résultats
│
├── results/                      # Résultats et sorties
│   ├── maps/                    # Cartes de rendement
│   ├── metrics/                 # Métriques de performance
│   └── shap/                    # Analyses SHAP
│
└── docs/                         # Documentation
    └── essai_complet.pdf        # Document de maîtrise complet
```

## Installation et utilisation

### Prérequis

- Python 3.11
- Ubuntu 24.04 (ou système Linux équivalent)
- Compte Google Earth Engine

### Installation

```bash
# Création de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

### Bibliothèques principales

```bash
pip install scikit-learn xgboost numpy pandas geopandas matplotlib seaborn torch shap rasterio plotly
```

### Exécution des modèles

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap

# Chargement des données prétraitées
df = pd.read_csv("data/dataset_preprocessed.csv")

# Séparation temporelle
train = df[df["year"] <= 2020]
test  = df[df["year"] > 2020]

features = ["ppt_mm", "tmax", "tmin", "tmean",
            "ndvi", "evi", "lai", "vv", "vh",
            "drainage", "crop_type_lag1"]

X_train, y_train = train[features], train["rendement"]
X_test,  y_test  = test[features],  test["rendement"]

# Entraînement XGBoost
model = xgb.XGBRegressor(
    learning_rate=0.05, max_depth=5, subsample=0.75,
    colsample_bytree=0.75, gamma=0.2,
    reg_alpha=0.3, reg_lambda=1.5,
    n_estimators=300, early_stopping_rounds=30,
    eval_metric="rmse"
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=False)

# Évaluation et interprétation SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

## Technologies utilisées

| Domaine | Outils |
|---------|--------|
| Programmation | Python 3.11, VS Code |
| ML / DL | scikit-learn, XGBoost, PyTorch |
| Données | NumPy, Pandas, GeoPandas, Rasterio |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Télédétection | Google Earth Engine, QGIS, PCI Catalyst |
| Gestion spatiale | Mergin Maps, ArcGIS Pro |
| Contrôle de version | GitHub |

## Limites identifiées

- **Taille du jeu de données** : 168 observations d'entraînement — la diversité agroclimatique de la Montérégie n'est que partiellement couverte
- **Résolution spatiale** : Sentinel-2 (10 m) et Landsat (30 m) ne capturent pas la variabilité intraparcellaire dans les zones à forte hétérogénéité
- **Couverture nuageuse** : Discontinuités temporelles pendant les phases phénologiques clés (levée, floraison, remplissage)
- **Variables agronomiques manquantes** : Densité de semis, type d'hybride, gestion de la fertilisation
- **SVM et TabResNet** : Performances insuffisantes dues au faible volume de données et à l'hétérogénéité spatiotemporelle

## Perspectives d'amélioration

### Court terme
- Enrichir le jeu de données au-delà de 168 observations
- Intégrer des séries temporelles Sentinel-2 plus denses sur l'ensemble du cycle cultural
- Ajouter des variables agronomiques détaillées (densité de semis, hybride, fertilisation)
- Exploiter des degrés-jours et cumuls de précipitations comme variables climatiques

### Moyen terme
- Modèles hybrides : XGB-NN, RF-DNN pour combiner stabilité et flexibilité
- Validation spatiale indépendante (spatial block cross-validation)
- Plateforme interactive de visualisation via GeoServer
- Intégration de séries radar Sentinel-1 à haute fréquence

### Long terme
- Imagerie drone multispectrale pour calibration locale
- Réseaux neuronaux spatio-temporels (CNN-LSTM) avec séries temporelles continues
- Extension à d'autres cultures (soya, blé)
- Système opérationnel d'aide à la décision en agriculture de précision

## Références clés

- Bolton, D. K., & Friedl, M. A. (2013). Forecasting crop yield using remotely sensed vegetation indices and crop phenology metrics. *Agricultural and Forest Meteorology*, 173, 74–84.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings KDD*, 785–794.
- Jeong, J. H., et al. (2016). Random Forests for Global and Regional Crop Yield Predictions. *PLOS ONE*, 11(6), e0156571.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.
- FADQ (2023). Assurance récolte — Rendements de référence. fadq.qc.ca
- Magagi, R., et al. (2013). Canadian Experiment for Soil Moisture in 2010 (CanEx-SM10). *IEEE TGRS*, 51(1), 347–363.

## Citation

```bibtex
@mastersthesis{sene2026,
  author    = {SENE, Ibrahima},
  title     = {Prédiction des rendements du maïs en Montérégie par
               l'intelligence artificielle et la télédétection},
  school    = {Université de Sherbrooke},
  department= {Département de géomatique appliquée},
  year      = {2026},
  month     = jan,
  note      = {Essai de maîtrise (M.Sc. en géomatique appliquée et télédétection)}
}
```

## Auteur et supervision

**Ibrahima SENE**  
Maître ès Sciences (M. Sc.) en géomatique appliquée et télédétection  
Département de géomatique appliquée — Université de Sherbrooke  
Janvier 2026

| Rôle | Professeur |
|------|-----------|
| Supervision | Mickaël Germain |
| Co-supervision | Ramata Magagi |
| Examinateur interne | Yacine Bouroubi |

## Remerciements

- La Financière agricole du Québec (FADQ) — données de rendement historiques
- Environnement et Changement climatique Canada (ECCC) — données climatiques
- Institut de recherche et de développement en agroenvironnement (IRDA) — données pédologiques
- Le personnel du Département de géomatique appliquée, Université de Sherbrooke

  
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
---

*Dernière mise à jour : Janvier 2026*
