import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Modèle XGBoost - Prédiction des rendements agricoles")
print("Intégration des indices de végétation et analyse spatiale")
print("=" * 80)

########################### CHARGEMENT ET PRÉPARATION DES DONNÉES ##################################

csv_path = (r'/home/snabraham6/#modele_deep_learning/data_model/data_final/data_final_enriched_climate.csv')
df = pd.read_csv(csv_path)

print(f"\nDonnées chargées: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Période couverte: {df['year'].min()} - {df['year'].max()}")
print(f"Rendement moyen: {df['yield_tpha'].mean():.2f} t/ha")

# Encodage des variables catégorielles
le_drainage = LabelEncoder()
le_station = LabelEncoder()
le_field = LabelEncoder()

df['drainage_encoded'] = le_drainage.fit_transform(df['drainage'].fillna('Unknown'))
df['station_encoded'] = le_station.fit_transform(df['station_name'].fillna('Unknown'))
df['field_encoded'] = le_field.fit_transform(df['Field'].fillna('Unknown'))

# Création de variables temporelles
df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
df['year_squared'] = df['year_normalized'] ** 2

########################## INTÉGRATION DES INDICES DE VÉGÉTATION ####################################
# Structure préparée pour NDVI, NDWI, EVI, LAI
# Si ces colonnes existent dans vos données, elles seront automatiquement intégrées
# Sinon, vous pouvez les ajouter via fusion avec un autre fichier

vegetation_indices = ['NDVI', 'NDWI', 'EVI', 'LAI']
available_indices = [col for col in vegetation_indices if col in df.columns]

if available_indices:
    print(f"\nIndices de végétation détectés: {', '.join(available_indices)}")
else:
    print("\nAucun indice de végétation détecté dans les données actuelles.")
    print("Pour les intégrer, ajoutez les colonnes NDVI, NDWI, EVI, LAI au CSV.")
    
    # Génération de données synthétiques pour démonstration (à remplacer par vos vraies données)
    print("\nGénération de données synthétiques pour démonstration...")
    np.random.seed(42)
    df['NDVI'] = np.random.uniform(0.3, 0.9, len(df))
    df['NDWI'] = np.random.uniform(-0.3, 0.3, len(df))
    df['EVI'] = np.random.uniform(0.2, 0.8, len(df))
    df['LAI'] = np.random.uniform(1.0, 6.0, len(df))
    available_indices = vegetation_indices

############################# SECTION 3: SÉLECTION DES FEATURES ##########################################

target = 'yield_tpha'
features_to_exclude = [
    'yield_kg_ha', 'yield_tpha', 'region', 'zone', 
    'Field', 'station_name', 'drainage'
]

numerical_features = [
    'g_pente_mo',
    'tmean', 'tmax', 'tmin', 'rain_mm', 'ppt_mm', 'snow_cm',
    'drainage_encoded', 'station_encoded', 'field_encoded'
] + available_indices

# Vérifier la disponibilité des features
available_features = [f for f in numerical_features if f in df.columns]
print(f"\nFeatures sélectionnées: {len(available_features)}")
for feat in available_features:
    print(f"  - {feat}")

# Préparation des matrices X et y
y = df[target].values
X = df[available_features].copy()

# Gestion des valeurs manquantes
print(f"\nGestion des valeurs manquantes...")
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Normalisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)

print(f"Matrice finale: {X_scaled.shape}")

# ----- SECTION 4: DIVISION TRAIN/TEST -----

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

print(f"\nÉchantillons d'entraînement: {len(X_train)}")
print(f"Échantillons de test: {len(X_test)}")

######################### CONFIGURATION ET ENTRAÎNEMENT DU MODÈLE ##############################
xgb_params = {
    'learning_rate': 0.03,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 42,
    'tree_method': 'hist'
}

print("\n" + "=" * 80)
print("ENTRAÎNEMENT DU MODÈLE XGBOOST")
print("=" * 80)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

evals = [(dtrain, 'train'), (dtest, 'test')]
evals_result = {}

model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50,
    evals_result=evals_result
)

################################ ÉVALUATION DU MODÈLE ########################################

y_pred_train = model.predict(dtrain)
y_pred_test = model.predict(dtest)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
rrmse_test = (rmse_test / y_test.mean()) * 100

print("\n" + "=" * 80)
print("MÉTRIQUES DE PERFORMANCE")
print("=" * 80)
print(f"\nEntraînement:")
print(f"  R² = {r2_train:.4f}")
print(f"  RMSE = {rmse_train:.3f} t/ha")

print(f"\nTest:")
print(f"  R² = {r2_test:.4f}")
print(f"  RMSE = {rmse_test:.3f} t/ha")
print(f"  MAE = {mae_test:.3f} t/ha")
print(f"  RRMSE = {rrmse_test:.2f}%")

###################################### VISUALISATIONS ###############################################

# FIGURE 1 : Courbe d'apprentissage (par itérations)
fig1 = plt.figure(figsize=(7, 5))
epochs = len(evals_result['train']['rmse'])
plt.plot(range(epochs), evals_result['train']['rmse'], label='Entraînement', linewidth=2)
plt.plot(range(epochs), evals_result['test']['rmse'], label='Test', linewidth=2)
plt.xlabel('Itérations', fontsize=11)
plt.ylabel('RMSE (t/ha)', fontsize=11)
plt.title('Courbe d\'apprentissage', fontsize=10, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/courbe_apprentissage.png', dpi=300, bbox_inches='tight')
print(f"\nGraphique sauvegardé: courbe_apprentissage.png")
plt.show()

# FIGURE 2 : Observé vs Prédit (Test)
fig2 = plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identité')
plt.xlabel('Rendement observé (t/ha)', fontsize=11)
plt.ylabel('Rendement prédit (t/ha)', fontsize=11)
plt.title(f'Test du modèle (R² = {r2_test:.3f})', fontsize=10, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/test_modele.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: test_modele.png")
plt.show()

# FIGURE 3 : Analyse des résidus (2 graphiques côte à côte)
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
residuals = y_test - y_pred_test

# Distribution des résidus
ax1 = axes[0]
ax1.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
ax1.axvline(0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Résidus (t/ha)', fontsize=11)
ax1.set_ylabel('Fréquence', fontsize=11)
ax1.set_title('Figure a : Distribution des résidus', fontsize=10, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Résidus vs Prédit
ax2 = axes[1]
ax2.scatter(y_pred_test, residuals, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Rendement prédit (t/ha)', fontsize=11)
ax2.set_ylabel('Résidus (t/ha)', fontsize=11)
ax2.set_title('Figure b : Analyse des résidus', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/analyse_residus.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: analyse_residus.png")
plt.show()

################################ IMPORTANCE DES FEATURES #######################################################

importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance (Gain)', fontsize=12)
plt.ylabel('Variables', fontsize=12)
plt.title('Les variables les plus importantes', fontsize=10, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/importance_features.png', dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé: importance_features.png")
plt.show()

print("\n" + "=" * 80)
print("LES VARIABLES LES PLUS IMPORTANTES")
print("=" * 80)
print(importance_df.head(10).to_string(index=False))

###################################### ANALYSE SHAP #######################################################

print("\n" + "=" * 80)
print("ANALYSE SHAP - INTERPRÉTABILITÉ DU MODÈLE")
print("=" * 80)

try:
    # Conversion du modèle pour SHAP
    booster = model
    booster.feature_names = list(X_train.columns)
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title('Impact des variables sur les prédictions (SHAP)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/shap_summary.png', dpi=300, bbox_inches='tight')
    print(f"\nGraphique SHAP sauvegardé: shap_summary.png")
    plt.show()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, max_display=15)
    plt.title('Importance SHAP des variables', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/shap_importance.png', dpi=300, bbox_inches='tight')
    print(f"Graphique SHAP sauvegardé: shap_importance.png")
    plt.show()
    
    # SHAP distribution plot détaillé
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('Distribution des impacts SHAP', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/shap_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Graphique SHAP sauvegardé: shap_distribution.png")
    plt.show()
    
    # Calcul des valeurs SHAP moyennes
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    print("\nTop 10 des impacts SHAP moyens:")
    for i, (feat, val) in enumerate(mean_abs_shap.head(10).items(), 1):
        print(f"  {i}. {feat}: {val:.4f}")
        
except Exception as e:
    print(f"\nProblème avec l'analyse SHAP: {str(e)}")
    print("Utilisation de l'importance native du modèle à la place.")
    
    # Alternative: utiliser l'importance native
    plt.figure(figsize=(10, 6))
    importance_vals = list(importance.values())[:15]
    importance_keys = list(importance.keys())[:15]
    
    plt.barh(range(len(importance_keys)), importance_vals)
    plt.yticks(range(len(importance_keys)), importance_keys)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.title('Importance des variables', fontsize=10, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/shap_alternative.png', dpi=300, bbox_inches='tight')
    print(f"Graphique d'importance sauvegardé: shap_alternative.png")
    plt.show()

####################################### CARTOGRAPHIE DES RÉSULTATS ###################################################

print("\n" + "=" * 80)
print("PRÉPARATION DE LA CARTOGRAPHIE")
print("=" * 80)

# Prédiction sur l'ensemble du dataset
X_all_scaled = pd.DataFrame(
    scaler.transform(imputer.transform(df[available_features])),
    columns=available_features
)
d_all = xgb.DMatrix(X_all_scaled)
df['yield_predicted'] = model.predict(d_all)
df['residual'] = df['yield_tpha'] - df['yield_predicted']
df['abs_error'] = np.abs(df['residual'])

# Sauvegarde des résultats
output_df = df[[
    'Field', 'year', 'station_name', 'drainage',
    'yield_tpha', 'yield_predicted', 'residual', 'abs_error'
] + available_indices].copy()

output_df.to_csv(r'/home/snabraham6/#modele_deep_learning/xgboost_model/data/predictions_rendements.csv', index=False)
print(f"\nRésultats sauvegardés: predictions_rendements.csv")

# Analyse par champ
field_stats = df.groupby('Field').agg({
    'yield_tpha': ['mean', 'std'],
    'yield_predicted': ['mean', 'std'],
    'abs_error': 'mean'
}).round(3)
field_stats.columns = ['Rendement_obs_moy', 'Rendement_obs_std', 
                       'Rendement_pred_moy', 'Rendement_pred_std', 
                       'Erreur_abs_moy']

print("\n" + "=" * 80)
print("STATISTIQUES PAR CHAMP")
print("=" * 80)
print(field_stats)

# Cartographie spatiale si coordonnées disponibles
if 'coord_x' in df.columns and 'coord_y' in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Rendements observés
    sc1 = axes[0].scatter(df['coord_x'], df['coord_y'], c=df['yield_tpha'], 
                          s=100, cmap='YlGn', edgecolors='black', linewidths=0.5)
    axes[0].set_title('Rendements observés (t/ha)', fontsize=10, fontweight='bold')
    plt.colorbar(sc1, ax=axes[0])
    
    # Rendements prédits
    sc2 = axes[1].scatter(df['coord_x'], df['coord_y'], c=df['yield_predicted'], 
                          s=100, cmap='YlGn', edgecolors='black', linewidths=0.5)
    axes[1].set_title('Rendements prédits (t/ha)', fontsize=10, fontweight='bold')
    plt.colorbar(sc2, ax=axes[1])
    
    # Erreurs absolues
    sc3 = axes[2].scatter(df['coord_x'], df['coord_y'], c=df['abs_error'], 
                          s=100, cmap='Reds', edgecolors='black', linewidths=0.5)
    axes[2].set_title('Erreurs absolues (t/ha)', fontsize=10, fontweight='bold')
    plt.colorbar(sc3, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('Coordonnée X', fontsize=10)
        ax.set_ylabel('Coordonnée Y', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/cartographie_spatiale.png', dpi=300, bbox_inches='tight')
    print(f"\nCartographie sauvegardée: cartographie_spatiale.png")
    plt.show()
else:
    print("\nCoordonnées spatiales non disponibles - cartographie basique générée")
    
    # Cartographie alternative par année et champ (2 lignes, 1 colonne)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Évolution temporelle par champ
    for field in df['Field'].unique():
        field_data = df[df['Field'] == field].sort_values('year')
        axes[0].plot(field_data['year'], field_data['yield_tpha'], 
                    marker='o', label=f'Obs {field}', alpha=0.7)
        axes[0].plot(field_data['year'], field_data['yield_predicted'], 
                    marker='s', linestyle='--', label=f'Pred {field}', alpha=0.7)
    
    axes[0].set_xlabel('Année', fontsize=10)
    axes[0].set_ylabel('Rendement (t/ha)', fontsize=10)
    axes[0].set_title('Figure a : Évolution temporelle des rendements', fontsize=10, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot des erreurs par champ
    field_errors = [df[df['Field'] == field]['abs_error'].values 
                    for field in df['Field'].unique()]
    axes[1].boxplot(field_errors, labels=df['Field'].unique())
    axes[1].set_xlabel('Champ', fontsize=11)
    axes[1].set_ylabel('Erreur absolue (t/ha)', fontsize=10)
    axes[1].set_title('Figure b : Distribution des erreurs par champ', fontsize=10, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/analyse_temporelle.png', dpi=300, bbox_inches='tight')
    print(f"Analyse temporelle sauvegardée: analyse_temporelle.png")
    plt.show()
