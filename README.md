# Prédiction des rendements du maïs en Montérégie par IA et télédétection

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Université de Sherbrooke](https://img.shields.io/badge/UdeS-Géomatique-green)](https://www.usherbrooke.ca/)
[![Google Earth Engine](https://img.shields.io/badge/GEE-Télédétection-orange)](https://earthengine.google.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Versioning-lightgrey)](https://github.com/seni2701/ESSAI_GAE724)

---

## Description

Ce projet développe une approche intégrée combinant la **télédétection multisources** et l'**intelligence artificielle** pour estimer le rendement du maïs-grain dans la région de la Montérégie au Québec, sur la période **2010–2023**. L'objectif est de concevoir des modèles capables de reproduire la variabilité spatiale et temporelle du rendement à partir de données satellitaires (Sentinel-1, Sentinel-2, Landsat), climatiques, pédologiques et de rotation culturale.

Il s'agit, à notre connaissance, de l'une des **premières applications combinant explicitement Sentinel-1, Sentinel-2, Landsat et des indicateurs de rotation culturale** dans un cadre d'apprentissage supervisé à l'échelle régionale en Montérégie.

---

## Contexte

La Montérégie représente près de **40 % des superficies de maïs-grain** du Québec — plus de **3,6 millions de tonnes** produites sur environ **355 700 hectares** (MAPAQ, 2023). Les rendements moyens régionaux varient entre **9 et 11 t/ha**, pouvant atteindre **12 t/ha** dans les conditions les plus favorables, avec un rendement de référence FADQ de **11,5 t/ha** sur la période 2010–2023 (min : 8,1 t/ha en 2011 ; max : 13,4 t/ha en 2017).

Dans un contexte de changements climatiques et de transition numérique en agriculture, l'anticipation fiable des rendements constitue un enjeu stratégique pour la sécurité alimentaire et les programmes d'assurance récolte (FADQ, MAPAQ).

---

## Objectifs

### Objectif général

Développer et comparer différents modèles d'IA afin de prédire le rendement du maïs en Montérégie à partir de données multisources combinant télédétection, variables climatiques, pédologiques et rotation culturale.

### Objectifs spécifiques

1. Identifier les variables agroclimatiques, pédologiques et spectrales les plus déterminantes du rendement du maïs à partir des données satellitaires
2. Comparer la performance des modèles d'apprentissage automatique et profond pour évaluer leur capacité à représenter la variabilité spatiale et temporelle des rendements
3. Analyser l'effet de la rotation maïs–soya sur la stabilité et la précision des prédictions de rendement à l'échelle régionale

---

## Méthodologie

### Site d'étude

L'étude couvre la **région administrative de la Montérégie** (sud du Québec), territoire d'environ 11 000 km² regroupant 14 MRC. Dix parcelles agricoles (F1–F10) ont été retenues autour de Beloeil, Saint-Hyacinthe et Saint-Bernard-de-Michaudville (secteurs FADQ 6-01, 6-02, 6-03 et 6-05), avec des séries de rendements continues sur 2010–2023.

| Groupe | Parcelles | Caractéristiques pédologiques |
|--------|-----------|-------------------------------|
| Rotation maïs–soya régulière | F1, F3, F5, F7, F9 | Alternance quasi-annuelle |
| Monoculture plus fréquente | F2, F4, F6, F8, F10 | Zones à forte productivité |
| Drainage imparfait à modérément bon | F1, F3, F4, F6 | Zones légèrement dépressionnaires |
| Drainage modérément bon à bon | F2, F5, F10 | Meilleures conditions de croissance |
| Drainage rapide à excessif | F7, F8, F9 | Sols sableux, risque de stress hydrique |

### Données mobilisées

| Catégorie | Sources | Variables |
|-----------|---------|-----------|
| **Satellitaires** | Sentinel-2, Sentinel-1 (SAR), Landsat 5/7/8 via GEE | NDVI, EVI, LAI, VV, VH, GLCM |
| **Climatiques** | ECCC, Agrométéo Québec, MELCCFP (2010–2023) | tmax, tmin, tmean, ppt_mm |
| **Pédologiques** | BDPQ / IRDA | Texture, drainage, matière organique, pente |
| **Rotation culturale** | AAFC (Annual Crop Inventory) via GEE | crop_type_lag1, is_corn, is_soy |
| **Rendements historiques** | FADQ (2010–2023) | Rendement observé par parcelle (t/ha) |

**Stations météorologiques** : Marieville (7024627), Montréal–Saint-Hubert (7027329), Sorel (7028200), Verchères (7028700).

### Indices spectraux calculés (Sentinel-2, 10 m)

| Indice | Formule | Référence |
|--------|---------|-----------|
| NDVI | (NIR − Red) / (NIR + Red) | Tucker (1979) |
| EVI | 2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1) | Huete et al. (2002) |
| LAI | Estimé via modèle biophysique Sentinel-2 | Fang et al. (2019) |
| VV / VH | Rétrodiffusion SAR Sentinel-1 | Veloso et al. (2017) |

Le masquage des nuages est effectué via les bandes **QA60**, **SCL** et l'algorithme `s2cloudless`. La collection `COPERNICUS/S2_SR_HARMONIZED` garantit la cohérence radiométrique sur l'ensemble de la période. Des **composites mensuels** (juin, juillet, août) ont été générés par parcelle pour couvrir les phases phénologiques clés (croissance végétative, floraison, remplissage des grains).

### Prétraitement des données

- **Valeurs manquantes** : imputation par `SimpleImputer` (moyenne) ; interpolation temporelle et validation croisée entre stations voisines pour les données climatiques
- **Normalisation** : `StandardScaler` + méthode min-max pour les variables continues
- **Encodage** : `LabelEncoder` pour les variables catégorielles
- **Variables de rotation** : `is_soy`, `is_corn`, `crop_type_lag1` générées après masquage dynamique maïs/soya dans GEE
- **Division temporelle** : 70 % entraînement (2010–2020 → **110 observations**) / 30 % test (2021–2023 → **30 observations**)

### Modèles testés et hyperparamètres

| Modèle | Algorithme | Paramètres clés |
|--------|------------|-----------------|
| **Random Forest (RF)** | Bagging (ensemble) | n_estimators=150, max_depth=10, max_samples=0.8, max_features=0.5, min_samples_split=10, min_samples_leaf=4 |
| **XGBoost (XGB)** | Gradient boosting séquentiel | learning_rate=0.05, max_depth=5, subsample=0.75, colsample_bytree=0.75, gamma=0.2, α=0.3 (L1), λ=1.5 (L2), 300 itérations, early stopping=30 |
| **SVM** | Noyau radial (RBF) | C=10, ε=0.1, gamma=scale |
| **TabResNet** | Réseau neuronal résiduel (PyTorch) | learning_rate=0.001, batch_size=32, 500 époques, Dropout p=0.2, Batch Normalization |

### Stratégies de validation

| Modèle | Protocole principal | Protocole complémentaire |
|--------|---------------------|--------------------------|
| RF, XGB, TabResNet | Validation croisée k-fold (5 plis) | LOYO (RF et XGB) |
| SVM | TimeSeriesSplit (ordre chronologique) | LOYO |

Le protocole **Leave-One-Year-Out (LOYO)** exclut à tour de rôle chaque année complète du jeu d'entraînement, constituant une contrainte plus exigeante que la validation croisée classique. TabResNet exclu du LOYO en raison d'une convergence précoce instable sur n=140 observations.

**Métriques** : R², RMSE (t/ha), MAE (t/ha), erreur relative (% du rendement moyen FADQ = 11,5 t/ha)

**Interprétabilité** : Analyse **SHAP** (SHapley Additive exPlanations) pour RF et XGB

---

## Résultats

### Comparaison des performances (ensemble de test, 2021–2023)

| Modèle | R² | RMSE (t/ha) | MAE (t/ha) | Erreur relative |
|--------|----|-------------|------------|-----------------|
| **XGBoost** | **0,780** | **0,638** | **0,411** | **5,6 %** |
| Random Forest | 0,509 | 0,954 | 0,716 | 8,3 % |
| TabResNet | 0,443 | 1,134 | 0,812 | 9,9 % |
| SVM | 0,191 | 1,367 | 0,939 | 11,9 % |

> XGBoost améliore la précision d'environ **33 % (RMSE)** et **43 % (MAE)** par rapport au Random Forest. L'écart entre XGB et SVM représente une réduction de l'erreur relative de l'ordre de **53 %**.

### Validation LOYO — estimations temporelles conservatrices

| Modèle | MAE LOYO par champ (t/ha) | RMSE LOYO par champ (t/ha) |
|--------|---------------------------|----------------------------|
| XGBoost | 1,30 – 1,48 | 1,65 – 1,77 |
| Random Forest | 1,42 – 1,62 | 1,71 – 1,86 |

> Ces valeurs, nettement supérieures à celles de la validation croisée, reflètent la difficulté à généraliser sur des **années climatiquement non observées** et constituent des estimations plus représentatives de la performance réelle en conditions opérationnelles.

### Variables les plus influentes — analyse SHAP (XGBoost)

| Rang | Variable | Rôle agronomique |
|------|----------|-----------------|
| 1 | `ppt_mm` — précipitations cumulées | Facteur dominant ; bilan hydrique du maïs à chaque stade |
| 2 | `tmax` — température maximale | Contrainte thermique (floraison, remplissage des grains) |
| 3 | `v6_v8_ndvi_entropy` — entropie NDVI | Hétérogénéité et structure du couvert végétal |
| 4 | `VV`, `VH` — rétrodiffusion Sentinel-1 | Structure du couvert et état hydrique du sol |
| 5 | `crop_type_lag1` — rotation culturale | Effet du précédent cultural sur la fertilité et structure du sol |
| 6 | `EVI`, `NDVI_max`, `LAI` | Vigueur végétative et biomasse |
| 7 | `tmean`, `tmin` | Conditions thermiques complémentaires |

### Observations clés

- XGBoost reproduit fidèlement les **années de stress climatique** (2011 : excès d'humidité → 8,1 t/ha ; 2014 et 2019 : déficits hydriques) et les **années à forte productivité** (2016, 2017, 2022 : >12,5 t/ha)
- En 2019, XGB prédit −3,8 t/ha vs −3,9 t/ha observé (FADQ) ; le RF sous-estime à −2,7 t/ha
- Les parcelles en **rotation maïs–soya** (F1, F3, F5, F7, F9) présentent des MAE médianes < 0,3 t/ha vs 0,5–0,7 t/ha pour les parcelles en monoculture
- Le **drainage du sol** est déterminant : 9,5 t/ha sur sols à drainage excessif (SRE) contre 12,0 t/ha sur sols à drainage subaquatique (SUB)
- La variabilité spatiale s'aligne avec les données FADQ : zones productives au centre/nord Montérégie (12–13 t/ha), moins productives au sud-ouest (<10 t/ha)
- Le RF tend à **sous-estimer les rendements élevés** (>12,5 t/ha) en raison de l'effet de moyennage propre au bagging

---

## Structure du dépôt

```
ESSAI_GAE724/
│
├── data/                         # Données d'entrée
│   ├── climate/                 # Données climatiques (ECCC, Agrométéo Québec)
│   ├── pedology/                # Données pédologiques (BDPQ / IRDA)
│   ├── satellite/               # Indices spectraux extraits via GEE
│   └── rotation/                # Rotation culturale AAFC (Annual Crop Inventory)
│
├── scripts/                      # Scripts de modélisation
│   ├── yield_rf.py              # Modèle Random Forest
│   ├── yield_svm.py             # Modèle SVM
│   ├── yield_xgboost.py         # Modèle XGBoost
│   └── modele_rendement_svm_FINAL.py
│
├── notebooks/                    # Notebooks d'analyse
│   ├── preprocessing.ipynb      # Prétraitement et préparation des données
│   └── visualization.ipynb      # Visualisations des résultats et SHAP
│
├── results/                      # Résultats et sorties
│   ├── maps/                    # Cartes de rendement prédit
│   ├── metrics/                 # Métriques de performance (R², RMSE, MAE)
│   └── shap/                    # Analyses SHAP (importance des variables)
│
└── docs/
    └── essai_complet.pdf        # Document de maîtrise complet (M.Sc., jan. 2026)
```

---

## Installation et utilisation

### Prérequis

- Python 3.11
- Ubuntu 24.04 ou Windows 11 (via miniconda / venv)
- Compte Google Earth Engine

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/seni2701/ESSAI_GAE724.git
cd ESSAI_GAE724

# Création de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# ou : venv\Scripts\activate    # Windows

# Installation des dépendances
pip install -r requirements.txt
```

### Bibliothèques principales

```bash
pip install scikit-learn xgboost torch numpy pandas geopandas \
            matplotlib seaborn plotly shap rasterio scipy scienceplots
```

### Exemple d'utilisation — XGBoost (modèle optimal)

```python
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chargement des données prétraitées
df = pd.read_csv("data/dataset_preprocessed.csv")

# Séparation temporelle : 70 % entraînement / 30 % test
train = df[df["year"] <= 2020]   # 110 observations
test  = df[df["year"] > 2020]    # 30 observations (2021–2023)

features = ["ppt_mm", "tmax", "tmin", "tmean",
            "ndvi", "evi", "lai", "vv", "vh",
            "drainage", "g_pente_mo", "crop_type_lag1"]

X_train, y_train = train[features], train["rendement"]
X_test,  y_test  = test[features],  test["rendement"]

# Entraînement XGBoost
# Résultats attendus : R²=0.780, RMSE=0.638 t/ha, MAE=0.411 t/ha
model = xgb.XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    subsample=0.75,
    colsample_bytree=0.75,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=1.5,
    n_estimators=300,
    early_stopping_rounds=30,
    eval_metric="rmse"
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# Évaluation
y_pred = model.predict(X_test)
rmse   = mean_squared_error(y_test, y_pred, squared=False)
print(f"R²          : {r2_score(y_test, y_pred):.3f}")
print(f"RMSE        : {rmse:.3f} t/ha")
print(f"MAE         : {mean_absolute_error(y_test, y_pred):.3f} t/ha")
print(f"Erreur rel. : {rmse / 11.5 * 100:.1f} %")

# Interprétation SHAP
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

---

## Technologies utilisées

| Domaine | Outils |
|---------|--------|
| Programmation | Python 3.11, VS Code |
| ML / DL | scikit-learn, XGBoost, PyTorch |
| Données | NumPy, Pandas, GeoPandas, Rasterio, SciPy |
| Visualisation | Matplotlib, Seaborn, Plotly, scienceplots |
| Télédétection | Google Earth Engine (GEE), QGIS |
| Gestion spatiale | Mergin Maps, ArcGIS Pro |
| Interprétabilité | SHAP |
| Contrôle de version | GitHub |

---

## Limites identifiées

- **Taille du jeu de données** : 140 observations (10 parcelles × 14 ans), dont 110 en entraînement et 30 en test. Les seuils de stabilité documentés sont de 200–500 observations pour RF/XGB, plusieurs centaines pour le SVM, et plusieurs milliers pour le TabResNet — tous les modèles opèrent en deçà de leur seuil optimal.
- **Ensemble de test réduit** : 30 observations (3 par champ, 2021–2023). Une seule année climatiquement atypique suffit à dégrader partiellement les métriques ; la validation LOYO fournit des estimations plus représentatives.
- **Résolution spatiale** : Sentinel-2 (10 m) et Landsat (30 m) ne capturent pas la variabilité intraparcellaire dans les zones à forte hétérogénéité pédologique.
- **Couverture nuageuse** : Discontinuités temporelles pendant les phases phénologiques clés (levée, floraison, remplissage des grains), atténuées par `s2cloudless` et la collection harmonisée Sentinel-2 SR.
- **Variables agronomiques absentes** : Densité de semis, type d'hybride, fertilisation azotée — leur intégration pourrait améliorer le R² de 10 à 20 points de pourcentage.
- **Rendement historique moyen par parcelle non intégré** : Disponible dans la base FADQ, il constitue l'un des prédicteurs les plus informatifs du rendement futur et une priorité pour les travaux ultérieurs.
- **Données terrain non intégrées** : La campagne de terrain de l'été 2025 (rugosité de surface, humidité du sol, biomasse) constitue une opportunité concrète pour calibrer les indices satellitaires et valider indépendamment les prédictions.
- **Absence de validation spatiale indépendante** : La généralisation à des parcelles hors du jeu d'entraînement reste à établir.

---

## Perspectives d'amélioration

### Court terme
- Intégrer les données terrain de l'été 2025 (rugosité, humidité du sol, biomasse) pour calibrer et valider les indices satellitaires
- Enrichir le jeu de données avec de nouvelles parcelles et des données FADQ additionnelles
- Ajouter le **rendement historique moyen par parcelle** comme variable d'entrée (priorité méthodologique)
- Intégrer des variables agronomiques détaillées (type d'hybride, densité de semis, fertilisation azotée)
- Exploiter des **degrés-jours** et **cumuls de précipitations** par stade phénologique

### Moyen terme
- Modèles hybrides : **XGB-NN**, **RF-DNN** pour combiner stabilité et flexibilité
- **Validation spatiale par blocs** (spatial block cross-validation) pour évaluer la généralisation hors-domaine
- Imagerie drone multispectrale pour calibration locale à l'échelle intraparcellaire
- Quantification des incertitudes prédictives par bootstrap
- Méthodes d'explicabilité locale complémentaires (**LIME**)

### Long terme
- Réseaux neuronaux spatio-temporels (**CNN-LSTM**) avec séries temporelles continues Sentinel-1/2
- **Apprentissage par transfert** depuis des modèles pré-entraînés (Corn Belt américain, Ontario)
- Extension à d'autres cultures stratégiques (soya, blé, canola)
- Plateforme interactive de visualisation spatiale (**GeoServer / WebGIS**) à destination de la FADQ et du MAPAQ

---

## Références clés

- Bolton, D. K., & Friedl, M. A. (2013). Forecasting crop yield using remotely sensed vegetation indices and crop phenology metrics. *Agricultural and Forest Meteorology*, 173, 74–84.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings KDD*, 785–794.
- Fang, H., et al. (2019). An Overview of Global Leaf Area Index (LAI). *Reviews of Geophysics*, 57(3), 739–799.
- Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS*, 34.
- Huber, F., et al. (2022). Extreme Gradient Boosting for yield estimation compared with Deep Learning approaches. *Computers and Electronics in Agriculture*, 202, 107346.
- Jeong, J. H., et al. (2016). Random Forests for Global and Regional Crop Yield Predictions. *PLOS ONE*, 11(6), e0156571.
- Kang, Y., et al. (2020). Comparative assessment of environmental variables and machine learning algorithms for maize yield prediction in the US Midwest. *Environmental Research Letters*, 15(6), 064005.
- Kamir, E., et al. (2020). Estimating wheat yields in Australia using climate records, satellite image time series and machine learning methods. *ISPRS Journal*, 160, 124–135.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.
- Shahhosseini, M., et al. (2020). Forecasting Corn Yield With Machine Learning Ensembles. *Frontiers in Plant Science*, 11.
- Veloso, A., et al. (2017). Understanding the temporal behavior of crops using Sentinel-1 and Sentinel-2-like data. *Remote Sensing of Environment*, 199, 415–426.
- FADQ (2023). Assurance récolte — Rendements de référence. fadq.qc.ca

---

## Remerciements

- **Financière agricole du Québec (FADQ)** — données de rendement historiques (2010–2023)
- **Environnement et Changement climatique Canada (ECCC)** — données climatiques quotidiennes
- **Agriculture et Agroalimentaire Canada (AAFC)** — données de rotation culturale (Annual Crop Inventory)
- **Institut de recherche et de développement en agroenvironnement (IRDA)** — données pédologiques
- **Agrométéo Québec** et **MELCCFP** — données climatiques complémentaires
- Le personnel du Département de géomatique appliquée, Université de Sherbrooke

---

*Dernière mise à jour : Janvier 2026*
