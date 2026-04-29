# Prédiction des rendements du maïs en Montérégie par IA et télédétection

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Université de Sherbrooke](https://img.shields.io/badge/UdeS-Géomatique-green)](https://www.usherbrooke.ca/)
[![Google Earth Engine](https://img.shields.io/badge/GEE-Télédétection-orange)](https://earthengine.google.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Versioning-lightgrey)](https://github.com/seni2701/ESSAI_GAE724)

---

## Description

Ce projet développe une approche intégrée combinant la **télédétection multisources** et l'**intelligence artificielle** pour estimer le rendement du maïs-grain dans la région de la Montérégie au Québec, sur la période **2010–2023**. L'objectif est de concevoir des modèles capables de reproduire la variabilité spatiale et temporelle du rendement à partir de données satellitaires (Sentinel-1, Sentinel-2, Landsat), climatiques, pédologiques et de rotation culturale.

Il s'agit, à notre connaissance, de l'une des **premières applications combinant explicitement Sentinel-1, Sentinel-2, Landsat et des indicateurs de rotation culturale** dans un cadre d'apprentissage supervisé à l'échelle régionale en Montérégie.

Essai de maîtrise (M.Sc.) présenté au 

**Département de géomatique appliquée**
**Faculté des lettres et sciences humaines**
**Université de Sherbrooke**
**Avril 2026**

**Jury d'évaluation**

| Rôle | Nom | Affiliation |
|------|-----|-------------|
| Directeur | Mickaël Germain | Département de géomatique appliquée, UdeS |
| Co-directrice | Ramata Magagi | Département de géomatique appliquée, UdeS |
| Examinateur interne | Yacine Bouroubi | Département de géomatique appliquée, UdeS |

---

## Contexte

La Montérégie représente près de **40 % des superficies de maïs-grain** du Québec — plus de **3,6 millions de tonnes** produites sur environ **355 700 hectares** (MAPAQ, 2023). Les rendements moyens régionaux varient entre **9 et 11 t/ha**, pouvant atteindre **12 t/ha** dans les conditions les plus favorables, avec un rendement de référence FADQ de **11,5 t/ha** sur la période 2010–2023 (min : 8,85 t/ha en 2011 ; max : 13,35 t/ha en 2017).

> La concentration est encore plus marquée en Montérégie-Ouest, où le maïs-grain peut représenter jusqu'à 55–60 % des superficies en grandes cultures certaines années (FADQ, 2023a).

Dans un contexte de changements climatiques et de transition numérique en agriculture, l'anticipation fiable des rendements constitue un enjeu stratégique pour la sécurité alimentaire et les programmes d'assurance récolte (FADQ, MAPAQ).

---

## Objectifs

### Objectif général

Développer et comparer différents modèles d'IA afin de prédire le rendement du maïs en Montérégie à partir de données multisources combinant télédétection, variables climatiques, pédologiques et rotation culturale.

### Objectifs spécifiques

1. Identifier les variables agroclimatiques, pédologiques et spectrales les plus déterminantes du rendement du maïs à partir des données satellitaires.
2. Comparer la performance des modèles d'apprentissage automatique et profond pour évaluer leur capacité à représenter la variabilité spatiale et temporelle des rendements.
3. Analyser l'effet de la rotation maïs–soya sur la stabilité et la précision des prédictions de rendement à l'échelle régionale.

---

## Méthodologie

### Site d'étude

L'étude couvre la **région administrative de la Montérégie** (sud du Québec), territoire d'environ 11 000 km² regroupant 14 MRC, avec des précipitations annuelles de 900 à 1 000 mm et une saison de croissance de mai à octobre. Dix parcelles agricoles (F1–F10) ont été retenues autour de Beloeil, Saint-Hyacinthe et Saint-Bernard-de-Michaudville (secteurs FADQ 6-01, 6-02, 6-03 et 6-05), avec des séries de rendements continues sur 2010–2023 et localisées via la plateforme Mergin Maps (format KML).

| Groupe | Parcelles | Caractéristiques pédologiques |
|--------|-----------|-------------------------------|
| Rotation maïs–soya régulière (quasi-annuelle) | F1, F3, F5, F7, F9 | Alternance marquée maïs–soya |
| Monoculture maïs plus fréquente | F2, F4, F6, F8, F10 | Zones à forte productivité |
| Drainage imparfait à modérément bon (DRI–LPI) | F1, F3, F4, F6 | Zones légèrement dépressionnaires |
| Drainage modérément bon à bon (MSU–SDM) | F2, F5, F10 | Meilleures conditions de croissance |
| Drainage rapide à excessif (SJU–SRE) | F7, F8, F9 | Sols sableux, risque de stress hydrique |

### Données mobilisées

| Catégorie | Sources | Variables |
|-----------|---------|-----------|
| **Satellitaires** | Sentinel-2 SR Harmonisé, Sentinel-1 (SAR), Landsat 5/7/8 via GEE | NDVI, EVI, LAI, VV, VH, GLCM |
| **Climatiques** | ECCC, Agrométéo Québec, MELCCFP (2010–2023) | tmax, tmin, tmean, ppt_mm |
| **Pédologiques** | BDPQ / IRDA / SISCan | Texture, drainage, matière organique, pente |
| **Rotation culturale** | AAFC (Annual Crop Inventory) via GEE | crop_type_lag1, is_monoculture, consec_corn |
| **Rendements historiques** | FADQ / MAPAQ (2010–2023) | Rendement observé par parcelle (t/ha) |

**Stations météorologiques** :

| Nom | Latitude (°N) | Longitude (°O) | Code ID | Source |
|-----|--------------|---------------|---------|--------|
| Marieville | 45,46 | −73,13 | 7024627 | ECCC |
| Montréal–Saint-Hubert | 45,52 | −73,42 | 7027329 | ECCC |
| Sorel | 46,03 | −73,12 | 7028200 | ECCC |
| Verchères | 45,78 | −73,35 | 7028700 | ECCC |

### Indices spectraux calculés (Sentinel-2, 10 m)

| Indice | Formule | Référence |
|--------|---------|-----------|
| NDVI | (NIR − Red) / (NIR + Red) | Tucker (1979) |
| EVI | 2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1) | Huete et al. (2002) |
| LAI | Estimé via NDVI ou modèle biophysique Sentinel-2 | Fang et al. (2019) |
| VV / VH | Rétrodiffusion SAR Sentinel-1 (après correction radiométrique et filtrage speckle) | Veloso et al. (2017) |

Le masquage des nuages a été effectué via les bandes **QA60**, **SCL** et l'algorithme `s2cloudless`. La collection `COPERNICUS/S2_SR_HARMONIZED` garantit la cohérence radiométrique sur l'ensemble de la période.

> En janvier 2022, l'ESA a introduit la version de traitement 04.00 de Sentinel-2, modifiant les valeurs de réflectance de surface dans plusieurs bandes spectrales avec des écarts pouvant atteindre ±5 %. Sans harmonisation, ces discontinuités radiométriques auraient introduit des biais systématiques dans les séries temporelles de NDVI et d'EVI calculées sur 2010–2023 (ESA, 2022).

Des **composites mensuels** (juin, juillet, août) ont été générés par parcelle pour couvrir les phases phénologiques clés (croissance végétative, floraison, remplissage des grains).

### Rotation culturale (2010–2023)

| Année | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 |
|-------|----|----|----|----|----|----|----|----|----|----|
| 2010 | Maïs | Maïs | Maïs | Soya | Maïs | Soya | Maïs | Maïs | Maïs | Maïs |
| 2011 | Soya | Maïs | Soya | Maïs | Soya | Maïs | Soya | Maïs | Soya | Maïs |
| 2012 | Maïs | Soya | Maïs | Maïs | Maïs | Maïs | Maïs | Soya | Maïs | Soya |
| 2013 | Soya | Maïs | Soya | Maïs | Soya | Maïs | Soya | Maïs | Soya | Maïs |
| 2014 | Maïs | Soya | Maïs | Soya | Maïs | Maïs | Maïs | Soya | Maïs | Maïs |
| 2015 | Soya | Maïs | Soya | Maïs | Soya | Soya | Soya | Maïs | Soya | Soya |
| 2016 | Maïs | Soya | Maïs | Soya | Maïs | Maïs | Maïs | Soya | Maïs | Maïs |
| 2017 | Maïs | Maïs | Maïs | Maïs | Maïs | Soya | Soya | Maïs | Maïs | Maïs |
| 2018 | Soya | Maïs | Soya | Maïs | Maïs | Maïs | Soya | Soya | Soya | Maïs |
| 2019 | Maïs | Soya | Maïs | Soya | Soya | Maïs | Maïs | Maïs | Maïs | Soya |
| 2020 | Soya | Maïs | Soya | Maïs | Maïs | Soya | Soya | Soya | Soya | Maïs |
| 2021 | Maïs | Maïs | Maïs | Soya | Maïs | Maïs | Maïs | Maïs | Maïs | Soya |
| 2022 | Maïs | Soya | Maïs | Maïs | Maïs | Soya | Maïs | Maïs | Maïs | Maïs |
| 2023 | Soya | Maïs | Soya | Maïs | Maïs | Maïs | Soya | Soya | Soya | Maïs |

> F1, F3, F5, F7, F9 : alternance maïs–soya quasi-annuelle. F2, F4, F6, F8, F10 : séquences de monoculture maïs plus fréquentes.

### Prétraitement des données

- **Valeurs manquantes** : imputation par `SimpleImputer` (moyenne) ; interpolation temporelle et validation croisée entre stations voisines pour les données climatiques
- **Normalisation** : `StandardScaler` pour les variables continues ; mise à l'échelle min-max (0, 1) appliquée en complément pour le TabResNet afin de satisfaire les contraintes de convergence des architectures neuronales profondes
- **Encodage** : `LabelEncoder` pour les variables catégorielles
- **Variables de rotation** : `crop_type_lag1`, `maïs_mono`, `maïs_rot` générées après masquage dynamique maïs/soya dans GEE
- **Division temporelle** : 70 % entraînement (2010–2020 → **110 observations**) / 30 % test (2021–2023 → **30 observations**)

### Modèles testés et hyperparamètres

| Modèle | Algorithme | Paramètres clés |
|--------|------------|-----------------|
| **Random Forest (RF)** | Bagging (ensemble) | n_estimators=150, max_depth=10, max_samples=0.8, max_features=0.5, min_samples_split=10, min_samples_leaf=4 |
| **XGBoost** | Gradient boosting séquentiel | learning_rate=0.05, max_depth=5, subsample=0.75, colsample_bytree=0.75, min_child_weight=5, gamma=0.2, α=0.3 (L1), λ=1.5 (L2), 300 itérations, early stopping=30 |
| **SVM** | Noyau radial (RBF) | C=10, ε=0.1, gamma=scale |
| **TabResNet** | Réseau neuronal résiduel (PyTorch) | learning_rate=0.001, batch_size=32, 500 époques, ReLU, Dropout p=0.2, Batch Normalization |

### Stratégies de validation

| Modèle | Protocole principal | Protocole complémentaire |
|--------|---------------------|--------------------------|
| RF, XGBoost | Validation croisée k-fold (5 plis, stratifiée) | LOYO |
| SVM | GridSearchCV 5 plis + GroupKFold par champ (TimeSeriesSplit) | — |
| TabResNet | GroupShuffleSplit 80/20 par champ | — |

Le protocole **Leave-One-Year-Out (LOYO)** exclut à tour de rôle chaque année complète du jeu d'entraînement, constituant une contrainte plus exigeante que la validation croisée classique. Le SVM et le TabResNet ont été exclus du LOYO en raison de l'instabilité des estimations documentée pour des effectifs inférieurs à leurs seuils de stabilité respectifs (plusieurs centaines d'observations pour le SVM selon Kamir et al., 2020 ; plusieurs milliers pour le TabResNet selon Gorishniy et al., 2021). Aucun test préliminaire empirique n'a été conduit pour vérifier cette instabilité, ce qui constitue une limite méthodologique à adresser dans les travaux ultérieurs.

> Une approche alternative aurait consisté à appliquer un sous-échantillonnage bootstrap sur les 130 observations disponibles à chaque itération LOYO pour le SVM et le TabResNet, afin de quantifier empiriquement le seuil d'instabilité propre à chaque architecture (He & Ma, 2013).

**Métriques** : R², RMSE (t/ha), MAE (t/ha), erreur relative (% du rendement moyen FADQ = 11,5 t/ha)

**Interprétabilité** : Analyse **SHAP** (SHapley Additive exPlanations) pour RF et XGBoost

---

## Résultats

### Comparaison des performances (validation croisée et ensemble de test 2021–2023)

| Modèle | R² (test) | RMSE (t/ha) | MAE (t/ha) | Erreur relative |
|--------|-----------|-------------|------------|-----------------|
| **XGBoost** | **0,785** | **0,631** | **0,408** | **5,5 %** |
| Random Forest | 0,545 | 0,919 | 0,667 | 8,0 % |
| TabResNet | 0,509 | 1,065 | 0,831 | 9,3 % |
| SVM | 0,180 | 1,376 | 0,993 | 12,0 % |

> XGBoost améliore la précision d'environ **31 % (RMSE)** et **39 % (MAE)** par rapport au Random Forest. L'écart entre XGBoost et SVM représente une réduction de l'erreur relative de l'ordre de **54 %**.

> **Note sur l'incertitude des métriques** : sur un ensemble de test de n = 30 observations, l'incertitude d'estimation du R² peut atteindre 0,10 à 0,15 point selon le degré d'hétéroscédasticité des résidus (Draper & Smith, 1998). Les valeurs du R² pour XGBoost (0,785) et RF (0,545) présentent une zone d'incertitude qui se chevauchent partiellement lorsque la dispersion des résidus est irrégulière entre les plages de rendement. Cette instabilité des métriques sur petit échantillon renforce la pertinence du recours à la validation LOYO, dont les 14 itérations indépendantes produisent des estimations d'erreur statistiquement plus robustes que la partition temporelle 70/30.

### Validation LOYO — estimations temporelles conservatrices

| Modèle | MAE LOYO par champ (t/ha) | RMSE LOYO par champ (t/ha) |
|--------|---------------------------|----------------------------|
| XGBoost | 1,08 – 1,75 | 1,23 – 2,01 |
| Random Forest | 1,15 – 2,08 | 1,45 – 2,30 |

> Ces valeurs, nettement supérieures à celles de la validation croisée, reflètent la difficulté à généraliser sur des **années climatiquement non observées** et constituent des estimations plus représentatives de la performance réelle en conditions opérationnelles. Le nombre d'années de maïs par champ varie de 7 (F3, F7) à 10 (F5, F10) selon la rotation.

### Synthèse de l'analyse des résidus — quatre modèles

| Caractéristique | XGBoost | RF | TabResNet | SVM |
|---|---|---|---|---|
| Moyenne des résidus (t/ha) | 0,06 | 0,09 | 0,20 | −0,19 |
| Plage principale des résidus (t/ha) | −1,0 à +1,0 | −1,5 à +1,5 | −1,0 à +2,0 | −1,0 à +1,0 |
| Résidus extrêmes observés (t/ha) | −2,1 ; +2,5 | −3,1 ; −2,6 | −2,3 ; +2,8 | +2,5 ; −4,0 |
| Biais directionnel | Faible surestimation | Faible surestimation | Surestimation modérée des faibles rendements | Sous-estimation marquée des rendements élevés |
| Hétéroscédasticité | Faible | Modérée | Modérée | Élevée |
| Concentration des extrêmes | 10–11 t/ha prédit | 11,5–12 t/ha prédit | 10–11 t/ha prédit | 12,0–12,5 t/ha prédit |

> **Lecture du biais directionnel** : un résidu positif (observé − prédit > 0) indique que le modèle prédit en dessous de la valeur observée (surestimation de l'erreur). Un résidu négatif indique que le modèle prédit au-dessus de la valeur observée (sous-estimation de l'observé, soit une prédiction trop haute). XGBoost et RF surestiment légèrement ; TabResNet surestime davantage sur les faibles rendements ; le SVM sous-estime structurellement les rendements élevés, avec des résidus négatifs atteignant −4,0 t/ha entre 12,0 et 12,5 t/ha prédits.

L'analyse comparative révèle que XGBoost présente la distribution de résidus la plus resserrée (RMSE = 0,63 t/ha, bande ±RMSE la plus étroite des quatre modèles) avec une quasi-symétrie en cloche confirmant l'absence de biais directionnel systématique. Le RF affiche une asymétrie à queue gauche prononcée (jusqu'à −3,1 t/ha) et une tendance décroissante des résidus reflétant la régression vers la moyenne propre au bagging. Le TabResNet présente le biais moyen le plus élevé (+0,20 t/ha) avec une queue droite s'étendant jusqu'à +3,0 t/ha, signe d'une surestimation concentrée sur les faibles rendements prédits. Le SVM affiche la dispersion la plus élevée (RMSE = 1,38 t/ha) et une distribution aplatie avec des résidus extrêmes négatifs localisés dans la plage 12,0–12,5 t/ha prédit, traduisant une incapacité structurelle à reproduire les rendements élevés.

### Variables les plus influentes — analyse SHAP (XGBoost)

| Rang | Variable | Rôle agronomique |
|------|----------|-----------------|
| 1 | `ppt_mm` — précipitations cumulées | Facteur dominant ; bilan hydrique du maïs à chaque stade phénologique |
| 2 | `tmax` — température maximale | Contrainte thermique (floraison, remplissage des grains) |
| 3 | `v6_v8_ndvi_entropy` — entropie NDVI | Hétérogénéité et structure du couvert végétal |
| 4 | `VV`, `VH` — rétrodiffusion Sentinel-1 | Structure du couvert et état hydrique du sol |
| 5 | `crop_type_lag1` — rotation culturale | Effet du précédent cultural sur la fertilité et structure du sol |
| 6 | `EVI`, `NDVI_max`, `LAI` | Vigueur végétative et biomasse |
| 7 | `tmean`, `tmin` | Conditions thermiques complémentaires |

### Observations clés

- XGBoost reproduit fidèlement les **années de stress climatique** (2011 : excès d'humidité, ~8,7 t/ha ; 2019 : conditions froides et pluvieuses) et les **années à forte productivité** (2016, 2017, 2022 : >12,5 t/ha)
- En 2019, XGBoost prédit −3,8 t/ha vs −3,9 t/ha observé (FADQ) ; le RF sous-estime à −2,7 t/ha
- Les parcelles en **rotation maïs–soya** (F1, F3, F5, F7, F9) présentent des MAE médianes < 0,3 t/ha vs 0,5–0,7 t/ha pour les parcelles en monoculture plus fréquente
- Le **drainage du sol** est déterminant : 9,5 t/ha sur sols à drainage excessif (SRE) contre 12,0 t/ha sur sols à drainage subaquatique (SUB)
- La variabilité spatiale s'aligne avec les données FADQ : zones productives au centre/nord Montérégie (12–13 t/ha), moins productives au sud-ouest (<10 t/ha)
- Le RF tend à **sous-estimer les rendements élevés** (>12,5 t/ha) en raison de l'effet de moyennage propre au bagging (Breiman, 2001)
- Le TabResNet converge dès l'époque 5 et compresse les prédictions vers la moyenne — stable mais peu discriminant sur les extrêmes
- Le SVM présente une hétéroscédasticité élevée et une pente de régression observé/prédit de 0,25

---

## Environnement de développement

| Outil | Rôle | Bibliothèques / notes |
|-------|------|-----------------------|
| Google Earth Engine (GEE) | Extraction, traitement et composites temporels Sentinel-2/1, Landsat 5/7/8 | Scripts JavaScript + Python API ; gee_map |
| Python 3.11 / VS Code | Modélisation IA/ML, traitement statistique, analyse spatio-temporelle | scikit-learn, torch, pandas, numpy, matplotlib, shap, seaborn, scienceplots, scipy |
| VS Code + GitHub | Gestion de versions, reproductibilité | GitHub Actions |
| Excel / CSV | Saisie, vérification, format pivot pour GEE et Python | — |
| ArcGIS Pro / Mergin Maps | Cartographie du site d'étude, collecte terrain | Outils SIG de base |

Environnement virtuel : miniconda/venv, Ubuntu 24.04 / Windows 11.

---

## Structure du dépôt

```
ESSAI_GAE724/
│
├── data/                         # Données d'entrée
│   ├── climate/                 # Données climatiques (ECCC, Agrométéo Québec, MELCCFP)
│   ├── pedology/                # Données pédologiques (BDPQ / IRDA / SISCan)
│   ├── satellite/               # Indices spectraux extraits via GEE
│   └── rotation/                # Rotation culturale AAFC (Annual Crop Inventory)
│
├── scripts/                      # Scripts de modélisation
│   ├── yield_rf.py              # Modèle Random Forest
│   ├── yield_svm.py             # Modèle SVM
│   ├── yield_xgboost.py         # Modèle XGBoost
│   └── yield_TabResNet.py       # Modèle TabResNet
│
├── notebooks/                    # Notebooks d'analyse
│   ├── preprocessing.ipynb      # Prétraitement et préparation des données
│   └── visualization.ipynb      # Visualisations des résultats et SHAP
│
├── results/                      # Résultats et sorties
│   ├── metrics/                 # Métriques de performance (R², RMSE, MAE)
│   └── shap/                    # Analyses SHAP (importance des variables)
│
└── docs/
    └── essai_GAE724_version_finale_IB.pdf   # Document de maîtrise complet (M.Sc., jan. 2026)
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
# Résultats attendus : R²=0.785, RMSE=0.631 t/ha, MAE=0.408 t/ha
model = xgb.XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=5,
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

- **Taille du jeu de données** : 140 observations (10 parcelles × 14 ans), dont 110 en entraînement et 30 en test. Les seuils de stabilité documentés sont de 200–500 observations pour RF/XGBoost, plusieurs centaines pour le SVM, et plusieurs milliers pour le TabResNet — tous les modèles opèrent en deçà de leur seuil optimal.
- **Incertitude des métriques sur petit échantillon** : sur n = 30 observations en test, l'incertitude d'estimation du R² peut atteindre 0,10 à 0,15 point selon le degré d'hétéroscédasticité des résidus. Les métriques doivent être interprétées comme des indicateurs comparatifs relatifs plutôt que comme des mesures absolues de généralisation (Draper & Smith, 1998).
- **Comparaison asymétrique des modèles** : le SVM et le TabResNet n'ont pas été soumis au protocole LOYO, rendant leur comparaison avec RF et XGBoost partiellement déséquilibrée. Aucun test empirique préliminaire n'a vérifié cette instabilité.
- **Résolution spatiale** : Sentinel-2 (10 m) et Landsat (30 m) ne capturent pas la variabilité intraparcellaire dans les zones à forte hétérogénéité pédologique.
- **Couverture nuageuse** : discontinuités temporelles pendant les phases phénologiques clés (levée, floraison, remplissage des grains), atténuées par `s2cloudless` et la collection harmonisée Sentinel-2 SR.
- **Variables agronomiques absentes** : densité de semis, type d'hybride, fertilisation azotée — leur intégration pourrait améliorer le R² de 10 à 20 points de pourcentage.
- **Rendement historique moyen par parcelle non intégré** : disponible dans la base FADQ, il constitue l'un des prédicteurs les plus informatifs du rendement futur et une priorité pour les travaux ultérieurs.
- **Données terrain non intégrées** : la campagne de terrain de l'été 2025 (rugosité de surface, humidité du sol, biomasse) constitue une opportunité concrète pour calibrer les indices satellitaires et valider indépendamment les prédictions.
- **Absence de validation spatiale indépendante** : la généralisation à des parcelles hors du jeu d'entraînement reste à établir.
- **Absence de carte de prédiction spatiale** : la production d'une carte de rendement prédit à l'échelle régionale reste à réaliser pour pleinement valoriser la dimension géospatiale de l'étude.

---

## Perspectives d'amélioration

### Court terme
- Intégrer les données terrain de l'été 2025 (rugosité, humidité du sol, biomasse) pour calibrer et valider les indices satellitaires
- Enrichir le jeu de données avec de nouvelles parcelles et des données FADQ additionnelles
- Ajouter le **rendement historique moyen par parcelle** comme variable d'entrée (priorité méthodologique)
- Intégrer des variables agronomiques détaillées (type d'hybride, densité de semis, fertilisation azotée) et des variables climatiques cumulatives (degrés-jours, cumuls par stade phénologique)
- Produire une **carte de rendement prédit** à l'échelle régionale pour compléter la dimension géospatiale de l'étude
- Exploiter les images drone multispectrales pour calibration locale à l'échelle intraparcellaire

### Moyen terme
- Modèles hybrides : **XGBoost-NN**, **RF-DNN** pour combiner stabilité et flexibilité
- **Validation spatiale par blocs** (spatial block cross-validation) pour évaluer la généralisation hors-domaine
- Quantification des incertitudes prédictives par bootstrap, notamment pour les seuils d'instabilité du SVM et du TabResNet en contexte LOYO
- Méthodes d'explicabilité locale complémentaires (**LIME**)
- Séries Sentinel-2 plus denses couvrant l'ensemble du cycle cultural

### Long terme
- Réseaux neuronaux spatio-temporels (**CNN-LSTM**) avec séries temporelles continues Sentinel-1/2
- **Apprentissage par transfert** depuis des modèles pré-entraînés (Corn Belt américain, Ontario)
- Extension à d'autres cultures stratégiques (soya, blé, canola)
- Plateforme interactive de visualisation spatiale (**GeoServer / WebGIS**) à destination de la FADQ et du MAPAQ

---

## Références clés

- Bolton, D. K., & Friedl, M. A. (2013). Forecasting crop yield using remotely sensed vegetation indices and crop phenology metrics. *Agricultural and Forest Meteorology*, 173, 74–84.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings KDD*, 785–794.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis* (3rd ed.). Wiley.
- Fang, H., et al. (2019). An Overview of Global Leaf Area Index (LAI). *Reviews of Geophysics*, 57(3), 739–799.
- Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS*, 34.
- He, H., & Ma, Y. (2013). *Imbalanced Learning: Foundations, Algorithms, and Applications*. Wiley-IEEE Press.
- Huber, F., et al. (2022). Extreme Gradient Boosting for yield estimation compared with Deep Learning approaches. *Computers and Electronics in Agriculture*, 202, 107346.
- Jeong, J. H., et al. (2016). Random Forests for Global and Regional Crop Yield Predictions. *PLOS ONE*, 11(6), e0156571.
- Kang, Y., et al. (2020). Comparative assessment of environmental variables and machine learning algorithms for maize yield prediction in the US Midwest. *Environmental Research Letters*, 15(6), 064005.
- Kamir, E., et al. (2020). Estimating wheat yields in Australia using climate records, satellite image time series and machine learning methods. *ISPRS Journal*, 160, 124–135.
- Khaki, S., & Wang, L. (2019). Crop Yield Prediction Using Deep Neural Networks. *Frontiers in Plant Science*, 10.
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30.
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- Shahhosseini, M., et al. (2020). Forecasting Corn Yield With Machine Learning Ensembles. *Frontiers in Plant Science*, 11.
- Veloso, A., et al. (2017). Understanding the temporal behavior of crops using Sentinel-1 and Sentinel-2-like data. *Remote Sensing of Environment*, 199, 415–426.
- FADQ (2023a). Assurance récolte — Rendements de référence. fadq.qc.ca

---

## Remerciements

- **Financière agricole du Québec (FADQ)** — données de rendement historiques (2010–2023)
- **Environnement et Changement climatique Canada (ECCC)** — données climatiques quotidiennes
- **Agriculture et Agroalimentaire Canada (AAFC)** — données de rotation culturale (Annual Crop Inventory)
- **Institut de recherche et de développement en agroenvironnement (IRDA)** — données pédologiques
- **Agrométéo Québec** et **MELCCFP** — données climatiques complémentaires
- Le corps professoral du Département de géomatique appliquée, Université de Sherbrooke, en particulier le Pr Mickaël Germain et la Pre Ramata Magagi

---

*Dernière mise à jour : Avril 2026*
