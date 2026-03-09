import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("=== Modèle SVM - Prédiction des rendements (adapté au modèle RF) ===\n")

# 
############################# Chargement des données ################################################"
# 
data_path = r'/home/snabraham6/#modele_deep_learning/data_model/data_final/data_final_enriched_climate.csv'
df = pd.read_csv(data_path)
print(f"Données chargées : {df.shape[0]} observations, {df.shape[1]} variables\n")

########################## Sélection dynamique des features #######################
# Variables dérivées simples
if {'tmean', 'tmax', 'tmin', 'rain_mm', 'ppt_mm'}.issubset(df.columns):
    df['tmean2'] = df['tmean'] ** 2
    df['temp_range'] = df['tmax'] - df['tmin']
    df['rain_anomaly'] = df['rain_mm'] - df['rain_mm'].mean()
    df['ppt_ratio'] = df['rain_mm'] / (df['ppt_mm'] + 1e-6)

feature_candidates = [
    'region', 'zone', 'year',
    'g_pente_mo', 'tmean', 'tmax', 'tmin',
    'rain_mm', 'ppt_mm', 'snow_cm',
    'NDVI', 'NDWI', 'EVI', 'LAI',
    'drainage_encoded', 'station_name_encoded',
    'tmean2', 'temp_range', 'rain_anomaly', 'ppt_ratio'
]
target_col = 'yield_tpha'

# Filtrage automatique : on garde seulement les colonnes existantes
available_features = [c for c in feature_candidates if c in df.columns]
missing_features = [c for c in feature_candidates if c not in df.columns]

if missing_features:
    print(f"[] Colonnes absentes ignorées : {missing_features}\n")

print(f"Variables retenues : {available_features}\n")

###################### Préparation des données ##################################

needed = available_features + [target_col, 'Field']
df_model = df[needed].copy()
df_model = df_model.dropna(subset=[target_col])

X = df_model[available_features]
y = df_model[target_col]
fields = df_model['Field']
years = df_model['year']

######################## Split temporel ###################################"
train_mask = years <= 2020
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
fields_train = fields[train_mask]

print(f"Données d'entraînement : {len(X_train)} | Test : {len(X_test)}\n")

############################### Pipeline et GridSearch ########################################
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.85)),        # Réduit de 0.95 → 0.85 (CRITIQUE)
    ('svm', SVR(kernel='rbf'))
])

param_grid = {
    'svm__C': [0.01, 0.05, 0.1, 0.5, 1],    # Encore plus faible : max 1 au lieu de 10 (CRITIQUE)
    'svm__gamma': ['scale', 0.0001, 0.001], # Noyau très large (CRITIQUE)
    'svm__epsilon': [0.5, 1.0, 1.5, 2.0]     # Marges très larges (CRITIQUE)
}

print("Optimisation GridSearchCV avec régularisation maximale...")
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"\nMeilleurs paramètres : {grid.best_params_}")
print(f"Meilleur score (CV)  : {grid.best_score_:.3f}\n")

############################## Évaluation du modèle ##################################################
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rrmse = rmse / y_true.mean() * 100
    return r2, mae, rmse, rrmse

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

r2_tr, mae_tr, rmse_tr, rrmse_tr = metrics(y_train, y_train_pred)
r2_te, mae_te, rmse_te, rrmse_te = metrics(y_test, y_test_pred)

print("=" * 60)
print("RÉSULTATS DU MODÈLE")
print("=" * 60)
print(f"\nEntraînement (2010–2020): R²={r2_tr:.3f} | MAE={mae_tr:.3f} | RMSE={rmse_tr:.3f} | RRMSE={rrmse_tr:.2f}%")
print(f"Test (2021–2023): R²={r2_te:.3f} | MAE={mae_te:.3f} | RMSE={rmse_te:.3f} | RRMSE={rrmse_te:.2f}%\n")

############################## Validation croisée par champ #####################################
cv = GroupKFold(n_splits=5)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, groups=fields_train, scoring='r2', n_jobs=-1)
print(f"Validation croisée (GroupKFold): R² moyen={cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")

################################# Visualisations simples ############################################
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(y_train, y_train_pred, alpha=0.6, label=f'Train R²={r2_tr:.3f}')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
axes[0].set_title("Entraînement")
axes[0].set_xlabel("Obs (t/ha)")
axes[0].set_ylabel("Prédit (t/ha)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].scatter(y_test, y_test_pred, alpha=0.7, label=f'Test R²={r2_te:.3f}', color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title("Test")
axes[1].set_xlabel("Obs (t/ha)")
axes[1].set_ylabel("Prédit (t/ha)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

######################## Figures supplémentaires d'analyse ##############################################

# Figure 1 : Analyse des résidus (distribution + scatter)
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogramme des résidus - Train
axes[0, 0].hist(residuals_train, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Résidus (t/ha)', fontsize=10)
axes[0, 0].set_ylabel('Fréquence', fontsize=10)
axes[0, 0].set_title(f'Distribution des résidus - Train\n(Moyenne: {residuals_train.mean():.3f}, Écart-type: {residuals_train.std():.3f})', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Histogramme des résidus - Test
axes[0, 1].hist(residuals_test, bins=15, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Résidus (t/ha)', fontsize=10)
axes[0, 1].set_ylabel('Fréquence', fontsize=10)
axes[0, 1].set_title(f'Distribution des résidus - Test\n(Moyenne: {residuals_test.mean():.3f}, Écart-type: {residuals_test.std():.3f})', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Résidus vs Prédictions - Train
axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.5, s=50)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Prédictions (t/ha)', fontsize=10)
axes[1, 0].set_ylabel('Résidus (t/ha)', fontsize=10)
axes[1, 0].set_title('Résidus vs Prédictions - Train', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Résidus vs Prédictions - Test
axes[1, 1].scatter(y_test_pred, residuals_test, alpha=0.6, s=60, color='orange')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prédictions (t/ha)', fontsize=10)
axes[1, 1].set_ylabel('Résidus (t/ha)', fontsize=10)
axes[1, 1].set_title('Résidus vs Prédictions - Test', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/svm_model/figures/Analyse des résidus.png', dpi=300, bbox_inches='tight')
print("✓ Graphique des prédictions sauvegardé")
plt.show()

# Figure 2 : Performance par année (Test uniquement)
test_df_temp = df_model[~train_mask].copy()
test_df_temp['prediction'] = y_test_pred
test_df_temp['year'] = years[~train_mask].values

yearly_performance = []
for yr in sorted(test_df_temp['year'].unique()):
    yr_data = test_df_temp[test_df_temp['year'] == yr]
    if len(yr_data) >= 2:
        r2_yr = r2_score(yr_data[target_col], yr_data['prediction'])
        rmse_yr = np.sqrt(mean_squared_error(yr_data[target_col], yr_data['prediction']))
        mae_yr = mean_absolute_error(yr_data[target_col], yr_data['prediction'])
        yearly_performance.append({
            'year': yr,
            'n_obs': len(yr_data),
            'R²': r2_yr,
            'RMSE': rmse_yr,
            'MAE': mae_yr
        })

if yearly_performance:
    df_yearly = pd.DataFrame(yearly_performance)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R² par année
    axes[0].bar(df_yearly['year'], df_yearly['R²'], color='navy', edgecolor='black', alpha=0.7)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Année', fontsize=11)
    axes[0].set_ylabel('R²', fontsize=11)
    axes[0].set_title('R² par année (Test)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (yr, r2_val) in enumerate(zip(df_yearly['year'], df_yearly['R²'])):
        axes[0].text(yr, r2_val + 0.02, f'{r2_val:.2f}', ha='center', fontsize=9)
    
    # RMSE par année
    axes[1].bar(df_yearly['year'], df_yearly['RMSE'], color='tomato', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Année', fontsize=11)
    axes[1].set_ylabel('RMSE (t/ha)', fontsize=11)
    axes[1].set_title('RMSE par année (Test)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (yr, rmse_val) in enumerate(zip(df_yearly['year'], df_yearly['RMSE'])):
        axes[1].text(yr, rmse_val + 0.05, f'{rmse_val:.2f}', ha='center', fontsize=9)
    
    # MAE par année
    axes[2].bar(df_yearly['year'], df_yearly['MAE'], color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Année', fontsize=11)
    axes[2].set_ylabel('MAE (t/ha)', fontsize=11)
    axes[2].set_title('MAE par année (Test)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, (yr, mae_val) in enumerate(zip(df_yearly['year'], df_yearly['MAE'])):
        axes[2].text(yr, mae_val + 0.05, f'{mae_val:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/svm_model/figures/Performance par année.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique des prédictions sauvegardé")
    plt.show()

# Figure 3 : Distributions comparées (Observations vs Prédictions)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot comparatif - Train
data_train_box = [y_train.values, y_train_pred]
axes[0].boxplot(data_train_box, labels=['Observations', 'Prédictions'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_ylabel('Rendement (t/ha)', fontsize=11)
axes[0].set_title('Distribution Train : Obs vs Prédictions', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].text(1, y_train.max() * 0.95, f'Obs: μ={y_train.mean():.2f}, σ={y_train.std():.2f}', fontsize=9)
axes[0].text(2, y_train.max() * 0.95, f'Pred: μ={y_train_pred.mean():.2f}, σ={y_train_pred.std():.2f}', fontsize=9)

# Boxplot comparatif - Test
data_test_box = [y_test.values, y_test_pred]
axes[1].boxplot(data_test_box, labels=['Observations', 'Prédictions'], patch_artist=True,
                boxprops=dict(facecolor='lightsalmon', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Rendement (t/ha)', fontsize=11)
axes[1].set_title('Distribution Test : Obs vs Prédictions', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].text(1, y_test.max() * 0.95, f'Obs: μ={y_test.mean():.2f}, σ={y_test.std():.2f}', fontsize=9)
axes[1].text(2, y_test.max() * 0.95, f'Pred: μ={y_test_pred.mean():.2f}, σ={y_test_pred.std():.2f}', fontsize=9)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/svm_model/figures/Distributions comparées.png', dpi=300, bbox_inches='tight')
print("✓ Graphique des prédictions sauvegardé")
plt.show()

########################################## Validation par champ ###########################################
print("=" * 60)
print("VALIDATION PAR CHAMP (Test)")
print("=" * 60)

test_df = df_model[~train_mask].copy()
test_df['prediction'] = y_test_pred

field_stats = []
for f in sorted(test_df['Field'].unique()):
    sub = test_df[test_df['Field'] == f]
    if len(sub) < 2:
        continue
    r2f = r2_score(sub[target_col], sub['prediction'])
    rmsef = np.sqrt(mean_squared_error(sub[target_col], sub['prediction']))
    maef = mean_absolute_error(sub[target_col], sub['prediction'])
    field_stats.append({'Field': f, 'n': len(sub), 'R²': f"{r2f:.3f}", 'RMSE': f"{rmsef:.2f}", 'MAE': f"{maef:.2f}"})

print(pd.DataFrame(field_stats).to_string(index=False))
print("\n=== Modèle SVM exécuté avec gestion automatique des colonnes manquantes ===")

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/svm_model/figures/VALIDATION PAR CHAMP.png', dpi=300, bbox_inches='tight')
print("✓ Graphique des prédictions sauvegardé")
plt.show()
