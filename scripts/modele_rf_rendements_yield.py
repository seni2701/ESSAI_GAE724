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
print("Intégration des indices de télédétection Sentinel-2 RÉELS\n")

########################## CHARGEMENT DES DONNÉES ##################################################################
data = pd.read_csv(r'/home/snabraham6/#modele_deep_learning/data_model/data_final/data_final_enriched_climate.csv')
print(f"Dimensions: {data.shape[0]} lignes x {data.shape[1]} colonnes")

# Statistiques de base
print(f"\nRendement moyen: {data['yield_tpha'].mean():.2f} t/ha (écart-type: {data['yield_tpha'].std():.2f})")
print(f"Période: {data['year'].min():.0f} - {data['year'].max():.0f}")

###################### INTÉGRATION DES INDICES SENTINEL-2 ###########################################################
print("\n" + "="*70)
print("INTÉGRATION INDICES SENTINEL-2")
print("="*70)

indices_path = r'/home/snabraham6/#modele_deep_learning/data_model/data_final/indices_vegetation_sentinel2.csv'

try:
    df_indices = pd.read_csv(indices_path)
    print(f"Chargé: {len(df_indices)} champs")
    
    # Nettoyage
    if 'year' in df_indices.columns:
        df_indices = df_indices.drop('year', axis=1)
    
    # Correction EVI (division par 100 si > 1)
    for col in ['EVI_moyen', 'EVI_max']:
        if col in df_indices.columns:
            df_indices[col] = df_indices[col].apply(lambda x: x/100 if (not pd.isna(x) and x > 1) else x)
    
    # Fusion
    data = data.merge(df_indices, on='Field', how='left')
    
    # Renommage
    data['NDVI'] = data['NDVI_moyen']
    data['NDWI'] = data['NDWI_moyen']
    data['EVI'] = data['EVI_moyen']
    data['LAI'] = data['LAI_moyen']
    
    print(f"Indices intégrés - Plages:")
    for idx in ['NDVI', 'NDWI', 'EVI', 'LAI']:
        valid = data[idx].dropna()
        print(f"  {idx}: [{valid.min():.3f}, {valid.max():.3f}]")
    
except Exception as e:
    print(f"ERREUR: {e}")
    print("Utilisation de données simulées")
    np.random.seed(42)
    data['NDVI'] = np.random.uniform(0.3, 0.9, len(data))
    data['NDWI'] = np.random.uniform(-0.2, 0.3, len(data))
    data['EVI'] = np.random.uniform(0.2, 0.8, len(data))
    data['LAI'] = np.random.uniform(1.0, 6.0, len(data))

################################## ENCODAGE DES VARIABLES CATÉGORIELLES ################################################
for var in ['drainage', 'station_name', 'Field']:
    if var in data.columns:
        le = LabelEncoder()
        data[f'{var}_encoded'] = le.fit_transform(data[var].astype(str))

############################## SÉLECTION DES FEATURES (variables) ###################################################
target = 'yield_tpha'
feature_columns = [
    'region', 'zone', 'g_pente_mo',
    'tmean', 'tmax', 'tmin', 'rain_mm', 'ppt_mm', 'snow_cm',
    'NDVI', 'NDWI', 'EVI', 'LAI',
    'drainage_encoded', 'station_name_encoded'
]
feature_columns = [col for col in feature_columns if col in data.columns]

X_raw = data[feature_columns].copy()
y = data[target].copy()

print(f"\n{len(feature_columns)} features sélectionnées")

##################################### PRÉTRAITEMENT #################################################################
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)

##################################### DIVISION TRAIN/TEST #######################################################
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

############################################ HYPERPARAMETRES DU MODELE ################################################
print("\n" + "="*70)
print("ENTRAÎNEMENT RANDOM FOREST")
print("="*70)

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

######################################### ÉVALUATION #######################################################################
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nRésultats:")
print(f"  Train: R² = {r2_train:.3f}")
print(f"  Test:  R² = {r2_test:.3f} | RMSE = {rmse_test:.3f} t/ha | MAE = {mae_test:.3f} t/ha")

# Validation croisée
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
print(f"  CV (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Vérification surapprentissage
if r2_train - r2_test > 0.15:
    print()

############################################### VISUALISATIONS ################################################

# FIGURE 1 : Courbe d'apprentissage
from sklearn.model_selection import learning_curve

fig1 = plt.figure(figsize=(7, 5))

# Calculer la courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X_train, y_train, 
    cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

# Convertir en RMSE positif et calculer les moyennes
train_rmse = -train_scores.mean(axis=1)
test_rmse = -test_scores.mean(axis=1)

plt.plot(train_sizes, train_rmse, label='Entraînement', linewidth=2, marker='o')
plt.plot(train_sizes, test_rmse, label='Validation', linewidth=2, marker='o')
plt.xlabel('Taille du dataset d\'entraînement', fontsize=11)
plt.ylabel('RMSE (t/ha)', fontsize=11)
plt.title('Courbe d\'apprentissage', fontsize=10, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/fig_rf/courbe_apprentissage.png', dpi=300)
print(f"\nGraphique sauvegardé: courbe_apprentissage.png")
plt.show()


# FIGURE 2 : Test uniquement
fig2 = plt.figure(figsize=(7, 5))
ax_test = plt.subplot(1, 1, 1)

# Prédictions vs Observations (Test uniquement)
ax_test.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='orange', edgecolors='k', linewidth=0.5)
lim = [y_test.min(), y_test.max()]
ax_test.plot(lim, lim, 'r--', linewidth=2)
r2 = r2_score(y_test, y_pred_test)
ax_test.set_xlabel('Observé (t/ha)')
ax_test.set_ylabel('Prédit (t/ha)')
ax_test.set_title(f'Test du modèle (R² {r2:.3f})')
ax_test.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/fig_rf/test_SENTINEL2.png', dpi=300)
plt.show()


# FIGURE 3 : Analyse des résidus
fig3 = plt.figure(figsize=(14, 5))
residuals_test = y_test - y_pred_test

ax_hist = plt.subplot(1, 2, 1)
ax_hist.hist(residuals_test, bins=20, edgecolor='black', alpha=0.7)
ax_hist.axvline(0, color='red', linestyle='--', linewidth=2)
ax_hist.set_xlabel('Résidus (t/ha)')
ax_hist.set_title('Figure a : Distribution des résidus')
ax_hist.grid(alpha=0.3)

ax_resid = plt.subplot(1, 2, 2)
ax_resid.scatter(y_pred_test, residuals_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
ax_resid.axhline(0, color='red', linestyle='--', linewidth=2)
ax_resid.set_xlabel('Prédit (t/ha)')
ax_resid.set_ylabel('Résidus (t/ha)')
ax_resid.set_title('Figure b : Résidus vs Prédictions')
ax_resid.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/fig_rf/Analyse des résidus.png', dpi=300)
plt.show()

################################################ IMPORTANCE DES VARIABLES ########################################################
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n Variables importantes:")
print(feature_importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Les variables les plus importants')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/fig_rf/importance_SENTINEL2.png', dpi=300)
plt.show()

######################################################## ANALYSE SHAP ###########################################################
print("\nCalcul SHAP...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
plt.title('Importance SHAP')
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/random_f_model/fig_rf/shap_SENTINEL2.png', dpi=300)
plt.show()

############################################ CORRÉLATIONS INDICES-RENDEMENT ###################################################
print("\nCorrélations indices-rendement:")
for idx in ['NDVI', 'NDWI', 'EVI', 'LAI']:
    if idx in data.columns:
        corr = data[[idx, target]].corr().iloc[0, 1]
        print(f"  {idx}: r = {corr:+.3f}")

######################################### EXPORT #################################################################
data['rendement_predit'] = rf_model.predict(X_scaled)
data['erreur_prediction'] = data[target] - data['rendement_predit']

results_df = data[['Field', 'year', 'yield_tpha', 'rendement_predit', 'erreur_prediction']].copy()
results_df.to_csv(r'/home/snabraham6/#modele_deep_learning/random_f_model/fig_rf/predictions_SENTINEL2.csv', index=False)
