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

################################### CHARGEMENT ET EXPLORATION DES DONNÉES ###########################################
data = pd.read_csv(r'/home/snabraham6/#modele_deep_learning/data_model/data_final/data_final_enriched_climate_with_radar.csv')
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

############################### PRÉPARATION DES INDICES DE TÉLÉDÉTECTION ###################################################
# Note: Ces indices seront ajoutés lorsque les données satellitaires seront disponibles
# Pour l'instant, on simule leur structure pour l'architecture du modèle

print("\n--- Préparation des indices de télédétection ---")
# Simulation temporaire des indices (à remplacer par vos vraies données)
np.random.seed(42)
data['NDVI'] = np.random.uniform(0.3, 0.9, len(data))  # Normalized Difference Vegetation Index
data['EVI'] = np.random.uniform(0.2, 0.8, len(data))   # Enhanced Vegetation Index
data['LAI'] = np.random.uniform(1.0, 6.0, len(data))   # Leaf Area Index

print("Indices simulés (à remplacer par données réelles):")
print("- NDVI: Indice de végétation normalisé")
print("- EVI: Indice de végétation amélioré")
print("- LAI: Indice de surface foliaire")

##################################ROTATION DES CULTURES#######################
print("\n" + "="*70)
print("INTÉGRATION ROTATION DES CULTURES AAFC")
print("="*70)

try:
    # Charger les features de rotation enrichies
    df_rotation = pd.read_csv(r'/home/snabraham6/#modele_deep_learning/data_model/data_final/rotation_cultures_features_no_irda.csv')
    
    print(f"Données de rotation chargées: {df_rotation.shape}")
    print(f"Colonnes disponibles: {df_rotation.columns.tolist()}")
    
    # Sélectionner les features de rotation pertinentes
    rotation_features = ['Field', 'year', 'crop_type', 'crop_type_lag1', 
                        'corn_monoculture', 'consecutive_corn_years',
                        'corn_after_soy', 'is_corn', 'is_soy']
    
    # Vérifier quelles colonnes existent
    available_rotation_features = [col for col in rotation_features if col in df_rotation.columns]
    df_rotation_subset = df_rotation[available_rotation_features].copy()
    
    # Fusionner avec les données principales sur Field ET year
    print(f"\nFusion sur 'Field' et 'year'...")
    data = data.merge(df_rotation_subset, on=['Field', 'year'], how='left')
    
    print(f"Après fusion: {data.shape}")
    
    # Afficher les statistiques de rotation
    if 'corn_monoculture' in data.columns:
        print(f"\nStatistiques de rotation:")
        print(f"  - Observations avec monoculture de maïs: {data['corn_monoculture'].sum()}")
        print(f"  - Observations avec rotation: {data['corn_after_soy'].sum() if 'corn_after_soy' in data.columns else 'N/A'}")
        print(f"  - Années consécutives max: {data['consecutive_corn_years'].max() if 'consecutive_corn_years' in data.columns else 'N/A'}")
        
        # Pourcentage de données manquantes pour rotation
        missing_pct = (data['corn_monoculture'].isna().sum() / len(data)) * 100
        print(f"  - Données manquantes rotation: {missing_pct:.1f}%")
    
except Exception as e:
    print(f"ERREUR lors du chargement de la rotation: {e}")
    print("Création de variables de rotation par défaut (0)")
    data['crop_type'] = 0
    data['crop_type_lag1'] = 0
    data['corn_monoculture'] = 0
    data['consecutive_corn_years'] = 0
    data['corn_after_soy'] = 0
    data['is_corn'] = 0
    data['is_soy'] = 0

########################### SÉLECTION ET PRÉPARATION DES FEATURES #################################
target = 'yield_tpha'

# Variables radar à intégrer
radar_features = [
    'remplissage_ndvi_var', 'levee_ndvi_var', 'levee_evi',
    'v6_v8_ndvi_entropy', 'vt_r1_vv', 'v6_v8_glcm_homogeneity',
    'vt_r1_ndvi_var', 'remplissage_vv', 'vt_r1_vh',
    'vt_r1_evi', 'levee_vh', 'ndvi_max',
    'levee_vv', 'v6_v8_vh', 'remplissage_ndvi_entropy',
    'levee_lai', 'vt_r1_ndvi_entropy',
    'vt_r1_vv_vh', 'levee_glcm_homogeneity'
]

# Liste des features à utiliser (incluant maintenant les features radar)
feature_columns = [
    'g_pente_mo',
    'tmean', 'tmax', 'tmin',  'ppt_mm',
    'NDVI', 'EVI', 'LAI',
    'drainage_encoded', 'station_name_encoded','crop_type', 'crop_type_lag1', 
    'corn_monoculture', 'consecutive_corn_years',
    'corn_after_soy', 'is_corn', 'is_soy'
] + radar_features  # Ajout des features radar

# Features de rotation
rotation_features_model = [
    'crop_type', 'crop_type_lag1', 
    'corn_monoculture', 'consecutive_corn_years',
    'corn_after_soy', 'is_corn', 'is_soy'
]

# Filtrer les colonnes existantes
feature_columns = [col for col in feature_columns if col in data.columns]

X_raw = data[feature_columns].copy()
y = data[target].copy()

print(f"\n--- Variables utilisées pour la prédiction ---")
print(f"Nombre de features: {len(feature_columns)}")
print(f"Features de base: {len(feature_columns) - len([f for f in radar_features if f in feature_columns])}")
print(f"Features radar: {len([f for f in radar_features if f in feature_columns])}")
print(f"Features complètes: {feature_columns}")

################################ GESTION DES VALEURS MANQUANTES ET NORMALISATION ####################################
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

########################### DIVISION DES DONNÉES ##############################
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

# Validation croisée
cv_scores = cross_val_score(
    rf_model, X_scaled, y, cv=5, 
    scoring='r2', n_jobs=-1
)
print(f"\nValidation croisée (5-fold):")
print(f"  R² moyen = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

########################### ÉVALUATION DU MODÈLE ##################################
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


# Métriques sur l'ensemble de test
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
rrmse_test = (rmse_test / y_test.mean()) * 100

print(f"\nPerformance sur le test:")
print(f"  R² = {r2_test:.3f}")
print(f"  RMSE = {rmse_test:.3f} t/ha")
print(f"  MAE = {mae_test:.3f} t/ha")


############################ VISUALISATIONS ############################################
print("\n--- Génération des visualisations ---")

# Diagramme de dispersion observé vs prédit
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

############################ IMPORTANCE DES VARIABLES ######################################
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

########################## ANALYSE SHAP ###################################
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

############################ CARTOGRAPHIE DES RENDEMENTS ###################################
print("\n--- Génération de la cartographie ---")

# Prédiction sur l'ensemble complet
data['rendement_predit'] = rf_model.predict(X_scaled)
data['erreur_prediction'] = data[target] - data['rendement_predit']

############################ GRAPHIQUE PAR CHAMP ###################################
print("\n--- Génération des graphiques par champ ---")

# Récupérer la liste unique des champs et filtrer uniquement F1 à F10
all_fields = sorted(data['Field'].unique())
unique_fields = [f for f in all_fields if f.startswith('F') and f[1:].isdigit() and 1 <= int(f[1:]) <= 10]
unique_fields = sorted(unique_fields, key=lambda x: int(x[1:]))  # Trier par numéro

n_fields = len(unique_fields)

print(f"Nombre de champs à afficher: {n_fields}")
print(f"Champs sélectionnés: {unique_fields}")

# Calculer R² global pour le titre
mask_f1_f10 = data['Field'].isin(unique_fields)
test_indices_f1_f10 = data[mask_f1_f10].index.intersection(y_test.index)
y_test_f1_f10 = y_test[test_indices_f1_f10]
y_pred_test_f1_f10 = rf_model.predict(X_scaled.loc[test_indices_f1_f10])
r2_global = r2_score(y_test_f1_f10, y_pred_test_f1_f10)

# Calculer la disposition de la grille (3 colonnes)
n_cols = 3
n_rows = (n_fields + n_cols - 1) // n_cols

print(f"Grille: {n_rows} lignes x {n_cols} colonnes")
print(f"R² global (F1-F10): {r2_global:.3f}")

# Créer la figure avec subplots
fig = plt.figure(figsize=(18, 6 * n_rows))

# Titre global de la figure
fig.suptitle(f'Test du modèle - Champs F1 à F10 (R² {r2_global:.3f})', 
             fontsize=16, fontweight='bold', y=0.995)

gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

# Graphiques individuels par champ
for idx, field in enumerate(unique_fields):
    # Calculer la position dans la grille
    row = idx // n_cols
    col = idx % n_cols
    
    print(f"Champ {field}: position [{row}, {col}]")
    
    ax = fig.add_subplot(gs[row, col])
    
    # Filtrer les données pour ce champ
    field_mask = data['Field'] == field
    field_data = data[field_mask].copy()
    
    # Séparer en train et test pour ce champ
    field_test_mask = field_data.index.isin(y_test.index)
    field_test_data = field_data[field_test_mask]
    
    if len(field_test_data) > 0:
        y_field_test = field_test_data['yield_tpha']
        y_field_pred = field_test_data['rendement_predit']
        
        # Calculer R² pour ce champ
        if len(y_field_test) > 1 and y_field_test.std() > 0:
            r2_field = r2_score(y_field_test, y_field_pred)
        else:
            r2_field = np.nan
        
        # Tracer les points
        ax.scatter(y_field_test, y_field_pred, alpha=0.7, s=60, 
                  edgecolors='k', linewidth=0.8, color='steelblue')
        
        # Ligne 1:1
        if len(y_field_test) > 0:
            min_val_field = min(y_field_test.min(), y_field_pred.min())
            max_val_field = max(y_field_test.max(), y_field_pred.max())
            ax.plot([min_val_field, max_val_field], [min_val_field, max_val_field], 
                   'r--', linewidth=1.5, label='Ligne 1:1')
        
        # Titre avec R²
        if not np.isnan(r2_field):
            ax.set_title(f'Champ {field} (R² {r2_field:.3f})', fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'Champ {field} (R² N/A)', fontsize=10, fontweight='bold')
        
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f'Champ {field}\nPas de données test', 
               ha='center', va='center', fontsize=10)
        ax.set_title(f'Champ {field}', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Observé (t/ha)', fontsize=9)
    ax.set_ylabel('Prédit (t/ha)', fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/figures/test_par_champ.png', 
            dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: test_par_champ.png")
plt.show()

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
