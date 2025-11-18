import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("Modèle Random Forest pour la prédiction des rendements agricoles")
print("Intégration des indices de télédétection et analyse spatiale\n")

# ========== 1. CHARGEMENT ET EXPLORATION DES DONNÉES ==========
data = pd.read_csv(r'/home/snabraham6/#modele_deep_learning/data_model/data_final/data_final_enriched_climate.csv')
print(f"Dimensions des données: {data.shape[0]} lignes x {data.shape[1]} colonnes")
print(f"\nAperçu des colonnes disponibles:")
print(data.columns.tolist())

# Statistiques descriptives de la variable cible
print(f"\n--- Statistiques des rendements ---")
print(f"Rendement moyen: {data['yield_tpha'].mean():.2f} t/ha")
print(f"Rendement médian: {data['yield_tpha'].median():.2f} t/ha")
print(f"Écart-type: {data['yield_tpha'].std():.2f} t/ha")
print(f"Min: {data['yield_tpha'].min():.2f} t/ha | Max: {data['yield_tpha'].max():.2f} t/ha")

# Distribution temporelle
print(f"\n--- Distribution temporelle ---")  
print(data['year'].value_counts().sort_index())

# ========== 2. PRÉPARATION DES INDICES DE TÉLÉDÉTECTION ==========
# Note: Ces indices seront ajoutés lorsque les données satellitaires seront disponibles
# Pour l'instant, on simule leur structure pour l'architecture du modèle

print("\n--- Préparation des indices de télédétection ---")
# Simulation temporaire des indices (à remplacer par vos vraies données)
np.random.seed(42)
data['NDVI'] = np.random.uniform(0.3, 0.9, len(data))  # Normalized Difference Vegetation Index
data['NDWI'] = np.random.uniform(-0.2, 0.3, len(data))  # Normalized Difference Water Index
data['EVI'] = np.random.uniform(0.2, 0.8, len(data))   # Enhanced Vegetation Index
data['LAI'] = np.random.uniform(1.0, 6.0, len(data))   # Leaf Area Index

print("Indices simulés (à remplacer par données réelles):")
print("- NDVI: Indice de végétation normalisé")
print("- NDWI: Indice d'eau normalisé")
print("- EVI: Indice de végétation amélioré")
print("- LAI: Indice de surface foliaire")

# ========== 3. ENCODAGE DES VARIABLES CATÉGORIELLES ==========
categorical_vars = ['drainage', 'station_name', 'Field']
label_encoders = {}

for var in categorical_vars:
    if var in data.columns:
        le = LabelEncoder()
        data[f'{var}_encoded'] = le.fit_transform(data[var].astype(str))
        label_encoders[var] = le

# ========== 4. SÉLECTION ET PRÉPARATION DES FEATURES ==========
target = 'yield_tpha'

# Liste des features à utiliser
feature_columns = [
    'region', 'zone',
    'g_pente_mo',
    'tmean', 'tmax', 'tmin', 
    'rain_mm', 'ppt_mm', 'snow_cm',
    'NDVI', 'NDWI', 'EVI', 'LAI',
    'drainage_encoded', 'station_name_encoded'
]

# Filtrer les colonnes existantes
feature_columns = [col for col in feature_columns if col in data.columns]

X_raw = data[feature_columns].copy()
y = data[target].copy()

print(f"\n--- Variables utilisées pour la prédiction ---")
print(f"Nombre de features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# ========== 5. GESTION DES VALEURS MANQUANTES ET NORMALISATION ==========
print("\n--- Traitement des valeurs manquantes ---")
missing_counts = X_raw.isnull().sum()
print(missing_counts[missing_counts > 0])

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_raw),
    columns=X_raw.columns,
    index=X_raw.index
)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)

# ========== 6. DIVISION DES DONNÉES ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

print(f"\n--- Division des données ---")
print(f"Ensemble d'entraînement: {len(X_train)} échantillons")
print(f"Ensemble de test: {len(X_test)} échantillons")

############################################### ENTRAÎNEMENT DU MODÈLE RANDOM FOREST #############################################
print("\n--- Entraînement du modèle Random Forest ---")

rf_model = RandomForestRegressor(
    n_estimators=150,           
    max_depth=10,                
    min_samples_split=10,       
    min_samples_leaf=4,         
    max_features=0.5,           
    max_samples=0.8,            
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print()

# ========== 8. ÉVALUATION DU MODÈLE ==========
print("\n--- Évaluation des performances ---")
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Métriques sur l'ensemble d'entraînement
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae_train = mean_absolute_error(y_train, y_pred_train)
rrmse_train = (rmse_train / y_train.mean()) * 100

print(f"\nPerformance sur l'entraînement:")
print(f"  R² = {r2_train:.3f}")
print(f"  RMSE = {rmse_train:.3f} t/ha")
print(f"  MAE = {mae_train:.3f} t/ha")
print(f"  RRMSE = {rrmse_train:.2f}%")

# Métriques sur l'ensemble de test
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
rrmse_test = (rmse_test / y_test.mean()) * 100

print(f"\nPerformance sur le test:")
print(f"  R² = {r2_test:.3f}")
print(f"  RMSE = {rmse_test:.3f} t/ha")
print(f"  MAE = {mae_test:.3f} t/ha")
print(f"  RRMSE = {rrmse_test:.2f}%")

# Validation croisée
cv_scores = cross_val_score(
    rf_model, X_scaled, y, cv=5, 
    scoring='r2', n_jobs=-1
)
print(f"\nValidation croisée (5-fold):")
print(f"  R² moyen = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ========== 9. VISUALISATIONS ==========
print("\n--- Génération des visualisations ---")

# 9.1 Diagramme de dispersion observé vs prédit
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ensemble d'entraînement
axes[0].scatter(y_train, y_pred_train, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
min_val = min(y_train.min(), y_pred_train.min())
max_val = max(y_train.max(), y_pred_train.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ligne 1:1')
axes[0].set_xlabel('Rendement observé (t/ha)', fontsize=11)
axes[0].set_ylabel('Rendement prédit (t/ha)', fontsize=11)
axes[0].set_title(f'Entraînement (R² = {r2_train:.3f})', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Ensemble de test
axes[1].scatter(y_test, y_pred_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5, color='orange')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ligne 1:1')
axes[1].set_xlabel('Rendement observé (t/ha)', fontsize=11)
axes[1].set_ylabel('Rendement prédit (t/ha)', fontsize=11)
axes[1].set_title(f'Test (R² = {r2_test:.3f})', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/dispersion_rendement.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: dispersion_rendement.png")
plt.show()

# 9.2 Résidus
residuals_test = y_test - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution des résidus
axes[0].hist(residuals_test, bins=20, edgecolor='black', alpha=0.7)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Résidus (t/ha)', fontsize=11)
axes[0].set_ylabel('Fréquence', fontsize=11)
axes[0].set_title('Distribution des résidus', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Résidus vs prédictions
axes[1].scatter(y_pred_test, residuals_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Rendement prédit (t/ha)', fontsize=11)
axes[1].set_ylabel('Résidus (t/ha)', fontsize=11)
axes[1].set_title('Résidus vs Prédictions', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/residus_analyse.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: residus_analyse.png")
plt.show()

# ========== 10. IMPORTANCE DES VARIABLES ==========
print("\n--- Importance des variables ---")

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 des variables les plus importantes:")
print(feature_importance_df.head(10).to_string(index=False))

# Visualisation de l'importance
plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=11)
plt.ylabel('Variables', fontsize=11)
plt.title('Importance des variables (Top 15)', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/importance_variables.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegardé: importance_variables.png")
plt.show()

# ========== 11. ANALYSE SHAP ==========
print("\n--- Analyse SHAP ---")
print("Calcul des valeurs SHAP en cours...")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Importance SHAP des variables', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/shap_importance.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: shap_importance.png")
plt.show()

# SHAP summary plot détaillé
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('Distribution des impacts SHAP', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/shap_distribution.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: shap_distribution.png")
plt.show()

# ========== 12. CARTOGRAPHIE DES RENDEMENTS ==========
print("\n--- Génération de la cartographie ---")

# Prédiction sur l'ensemble complet
data['rendement_predit'] = rf_model.predict(X_scaled)
data['erreur_prediction'] = data[target] - data['rendement_predit']

# Carte des rendements observés par région/zone
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
pivot_obs = data.groupby(['region', 'zone'])['yield_tpha'].mean().reset_index()
scatter1 = plt.scatter(pivot_obs['region'], pivot_obs['zone'], 
                       c=pivot_obs['yield_tpha'], s=200, 
                       cmap='RdYlGn', edgecolors='black', linewidth=1.5)
plt.colorbar(scatter1, label='Rendement (t/ha)')
plt.xlabel('Région', fontsize=11)
plt.ylabel('Zone', fontsize=11)
plt.title('Rendements observés moyens', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
pivot_pred = data.groupby(['region', 'zone'])['rendement_predit'].mean().reset_index()
scatter2 = plt.scatter(pivot_pred['region'], pivot_pred['zone'], 
                       c=pivot_pred['rendement_predit'], s=200, 
                       cmap='RdYlGn', edgecolors='black', linewidth=1.5)
plt.colorbar(scatter2, label='Rendement (t/ha)')
plt.xlabel('Région', fontsize=11)
plt.ylabel('Zone', fontsize=11)
plt.title('Rendements prédits moyens', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/cartographie_rendements.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: cartographie_rendements.png")
plt.show()

# Carte des erreurs de prédiction
plt.figure(figsize=(10, 6))
pivot_err = data.groupby(['region', 'zone'])['erreur_prediction'].mean().reset_index()
scatter3 = plt.scatter(pivot_err['region'], pivot_err['zone'], 
                      c=pivot_err['erreur_prediction'], s=200, 
                      cmap='RdBu_r', edgecolors='black', linewidth=1.5,
                      vmin=-2, vmax=2)
plt.colorbar(scatter3, label='Erreur (t/ha)')
plt.xlabel('Région', fontsize=11)
plt.ylabel('Zone', fontsize=11)
plt.title('Erreurs de prédiction moyennes par zone', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/carte_erreurs.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: carte_erreurs.png")
plt.show()

# ========== 13. ANALYSE TEMPORELLE ==========
print("\n--- Analyse temporelle ---")

temporal_analysis = data.groupby('year').agg({
    'yield_tpha': 'mean',
    'rendement_predit': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))
plt.plot(temporal_analysis['year'], temporal_analysis['yield_tpha'], 
         marker='o', linewidth=2, label='Rendement observé', color='steelblue')
plt.plot(temporal_analysis['year'], temporal_analysis['rendement_predit'], 
         marker='s', linewidth=2, label='Rendement prédit', color='coral')
plt.xlabel('Année', fontsize=11)
plt.ylabel('Rendement moyen (t/ha)', fontsize=11)
plt.title('Evolution temporelle des rendements', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/evolution_temporelle.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: evolution_temporelle.png")
plt.show()

# ========== 14. EXPORT DES RÉSULTATS ==========
print("\n--- Export des résultats ---")

results_df = data[['region', 'zone', 'year', 'Field', 'yield_tpha', 
                   'rendement_predit', 'erreur_prediction']].copy()
results_df.to_csv(r'/home/snabraham6/#modele_deep_learning/random_f_model/données/predictions_rendements.csv', index=False)
print("Fichier sauvegardé: predictions_rendements.csv")

# Rapport de synthèse
summary_stats = {
    'Nombre_observations': len(data),
    'R2_test': r2_test,
    'RMSE_test': rmse_test,
    'MAE_test': mae_test,
    'RRMSE_test': rrmse_test,
    'R2_CV_mean': cv_scores.mean(),
    'R2_CV_std': cv_scores.std()
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv(r'/home/snabraham6/#modele_deep_learning/random_f_model/données/resume_modele.csv', index=False)
print("Fichier sauvegardé: resume_modele.csv")

print("\n" + "="*70)
print("ANALYSE TERMINÉE AVEC SUCCÈS")
print("="*70)
print("\nFichiers générés dans /mnt/user-data/outputs/:")
print("  - dispersion_rendement.png")
print("  - residus_analyse.png")
print("  - importance_variables.png")
print("  - shap_importance.png")
print("  - shap_distribution.png")
print("  - cartographie_rendements.png")
print("  - carte_erreurs.png")
print("  - evolution_temporelle.png")
print("  - predictions_rendements.csv")
print("  - resume_modele.csv")
