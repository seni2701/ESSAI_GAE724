# Changelog — ESSAI_GAE724

## [v1.0.0] — 2026
### Ajouté
- Extraction des indices spectraux via Google Earth Engine
  (NDVI, EVI, LAI, VV, VH)
- Sélection automatique Sentinel-1/2 et Landsat par année
- Pipeline ML complet : XGBoost, Random Forest, SVM, TabResNet
- GridSearchCV pour l'optimisation des hyperparamètres
- Analyse SHAP pour l'interprétabilité des modèles
- Export CSV des features temporelles (2010–2023)

## [À venir — v1.1.0]
- Intégration données pédologiques SoilGrids
- Validation croisée spatiale (Leave-One-Out par région)
- Interface de prédiction interactive (Streamlit)
