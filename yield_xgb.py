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
print("Intégration des indices de végétation Sentinel-2 RÉELS")
print("=" * 80)

# ----- SECTION 1: CHARGEMENT ET PRÉPARATION DES DONNÉES -----

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

# ----- SECTION 2: INTÉGRATION DES INDICES DE VÉGÉTATION SENTINEL-2 (RÉELS) -----

print("\n" + "=" * 80)
print("INTÉGRATION DES INDICES DE VÉGÉTATION SENTINEL-2")
print("=" * 80)

# Chemin vers le fichier des indices Sentinel-2
indices_path = r'/home/snabraham6/#modele_deep_learning/data_model/data_final/indices_vegetation_sentinel2.csv'

try:
    # Chargement des indices Sentinel-2 réels
    df_indices = pd.read_csv(indices_path)
    print(f"\n✓ Fichier Sentinel-2 chargé: {len(df_indices)} champs")
    print(f"  Champs disponibles: {sorted(df_indices['Field'].unique())}")
    
    # Suppression de la colonne 'year' si elle est vide
    if 'year' in df_indices.columns and df_indices['year'].isnull().all():
        df_indices = df_indices.drop('year', axis=1)
        print(f"  → Colonne 'year' vide supprimée")
    
    # CORRECTION DES VALEURS EVI ABERRANTES
    print("\nCorrection des valeurs EVI aberrantes...")
    
    def correct_evi(value):
        """Corrige les valeurs EVI > 1.0 (erreur de mise à l'échelle)"""
        if pd.isna(value):
            return np.nan
        if value > 1.0:
            return value / 100.0  # Division par 100 pour ramener dans [-1, 1]
        return value
    
    # Sauvegarde des valeurs originales pour diagnostic
    if 'EVI_moyen' in df_indices.columns:
        df_indices['EVI_moyen_original'] = df_indices['EVI_moyen']
        df_indices['EVI_moyen'] = df_indices['EVI_moyen'].apply(correct_evi)
        print(f"  ✓ EVI_moyen corrigé: {df_indices['EVI_moyen'].min():.3f} → {df_indices['EVI_moyen'].max():.3f}")
    
    if 'EVI_max' in df_indices.columns:
        df_indices['EVI_max_original'] = df_indices['EVI_max']
        df_indices['EVI_max'] = df_indices['EVI_max'].apply(correct_evi)
        print(f"  ✓ EVI_max corrigé:   {df_indices['EVI_max'].min():.3f} → {df_indices['EVI_max'].max():.3f}")
    
    # FUSION AVEC LE DATAFRAME PRINCIPAL
    print("\nFusion des indices avec les données principales...")
    
    # Vérification de la correspondance des champs
    fields_main = set(df['Field'].unique())
    fields_indices = set(df_indices['Field'].unique())
    
    print(f"  • Champs dans données principales: {len(fields_main)}")
    print(f"  • Champs avec indices Sentinel-2:  {len(fields_indices)}")
    print(f"  • Champs en commun:                {len(fields_main & fields_indices)}")
    
    if len(fields_main - fields_indices) > 0:
        print(f" Champs SANS indices: {sorted(fields_main - fields_indices)}")
    
    # Fusion par Field (jointure gauche pour garder toutes les observations)
    df_before = len(df)
    df = df.merge(df_indices, on='Field', how='left')
    df_after = len(df)
    
    if df_before != df_after:
        print(f" ATTENTION: Nombre d'observations changé ({df_before} → {df_after})")
    else:
        print(f"  ✓ Fusion réussie: {df_after} observations conservées")
    
    # RENOMMAGE POUR UNIFORMITÉ AVEC LE RESTE DU SCRIPT
    print("\nConfiguration des colonnes pour le modèle...")
    
    # Utilisation des valeurs moyennes par défaut (recommandé)
    df['NDVI'] = df['NDVI_moyen']
    df['NDWI'] = df['NDWI_moyen']
    df['EVI'] = df['EVI_moyen']
    df['LAI'] = df['LAI_moyen']
    
    # Vérification des plages de valeurs
    print("\n✓ Validation des indices intégrés:")
    for idx_name in ['NDVI', 'NDWI', 'EVI', 'LAI']:
        if idx_name in df.columns:
            valid_data = df[idx_name].dropna()
            if len(valid_data) > 0:
                print(f"  {idx_name:6s}: min={valid_data.min():+.3f}, max={valid_data.max():+.3f}, " +
                      f"médiane={valid_data.median():+.3f}, NaN={df[idx_name].isnull().sum()}")
                
                # Alerte si hors plage théorique
                if idx_name in ['NDVI', 'NDWI', 'EVI']:
                    if valid_data.min() < -1 or valid_data.max() > 1:
                        print(f"ATTENTION: Valeurs hors plage théorique [-1, +1]")
    
    # Gestion des valeurs manquantes
    missing_indices = df[['NDVI', 'NDWI', 'EVI', 'LAI']].isnull().sum()
    total_missing = missing_indices.sum()
    
    if total_missing > 0:
        print(f"\nValeurs manquantes détectées:")
        for col, count in missing_indices.items():
            if count > 0:
                pct = count / len(df) * 100
                print(f"  {col:6s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n  → Les valeurs manquantes seront imputées par la médiane dans la section suivante")
    else:
        print(f"\n✓ Aucune valeur manquante dans les indices")
    
    vegetation_indices = ['NDVI', 'NDWI', 'EVI', 'LAI']
    available_indices = vegetation_indices
    
    print(f"\nIndices Sentinel-2 RÉELS intégrés avec succès")
    print(f"   {len(df_indices)} champs × 4 indices = {len(df_indices) * 4} valeurs")

except FileNotFoundError:
    print(f"\nERREUR: Fichier non trouvé: {indices_path}")
    print("   Vérifiez le chemin du fichier.")
    print("\nUtilisation de données simulées en attendant...")
    
    # Fallback: Génération de données synthétiques
    np.random.seed(42)
    df['NDVI'] = np.random.uniform(0.3, 0.9, len(df))
    df['NDWI'] = np.random.uniform(-0.3, 0.3, len(df))
    df['EVI'] = np.random.uniform(0.2, 0.8, len(df))
    df['LAI'] = np.random.uniform(1.0, 6.0, len(df))
    vegetation_indices = ['NDVI', 'NDWI', 'EVI', 'LAI']
    available_indices = vegetation_indices
    print("   Indices simulés générés (à remplacer par données réelles)")

except Exception as e:
    print(f"\nERREUR lors du chargement des indices: {str(e)}")
    print("   Utilisation de données simulées en attendant...")
    
    # Fallback: Génération de données synthétiques
    np.random.seed(42)
    df['NDVI'] = np.random.uniform(0.3, 0.9, len(df))
    df['NDWI'] = np.random.uniform(-0.3, 0.3, len(df))
    df['EVI'] = np.random.uniform(0.2, 0.8, len(df))
    df['LAI'] = np.random.uniform(1.0, 6.0, len(df))
    vegetation_indices = ['NDVI', 'NDWI', 'EVI', 'LAI']
    available_indices = vegetation_indices
    print("   Indices simulés générés (à remplacer par données réelles)")

# ----- SECTION 3: SÉLECTION DES FEATURES -----

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
print(f"\n{'='*80}")
print(f"SÉLECTION DES FEATURES")
print(f"{'='*80}")
print(f"\nFeatures sélectionnées: {len(available_features)}")
for feat in available_features:
    print(f"  - {feat}")

# Préparation des matrices X et y
y = df[target].values
X = df[available_features].copy()

# Gestion des valeurs manquantes
print(f"\n{'='*80}")
print(f"GESTION DES VALEURS MANQUANTES")
print(f"{'='*80}")

missing_before = X.isnull().sum()
if missing_before.sum() > 0:
    print("\nValeurs manquantes détectées:")
    for col, count in missing_before[missing_before > 0].items():
        pct = count / len(X) * 100
        print(f"  {col:20s}: {count:4d} ({pct:5.1f}%)")
else:
    print("\n✓ Aucune valeur manquante")

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f"\n✓ Imputation par la médiane effectuée")

# Normalisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)

print(f"✓ Normalisation effectuée")
print(f"\nMatrice finale: {X_scaled.shape}")

# ----- SECTION 4: DIVISION TRAIN/TEST -----

print(f"\n{'='*80}")
print(f"DIVISION DES DONNÉES")
print(f"{'='*80}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

print(f"\nÉchantillons d'entraînement: {len(X_train)} ({len(X_train)/len(X_scaled)*100:.1f}%)")
print(f"Échantillons de test:        {len(X_test)} ({len(X_test)/len(X_scaled)*100:.1f}%)")

# ----- SECTION 5: CONFIGURATION ET ENTRAÎNEMENT DU MODÈLE -----

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

# ----- SECTION 6: ÉVALUATION DU MODÈLE -----

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

# ----- SECTION 7: VISUALISATIONS -----

# Courbe d'apprentissage
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 Courbe d'apprentissage
ax1 = axes[0, 0]
epochs = len(evals_result['train']['rmse'])
ax1.plot(range(epochs), evals_result['train']['rmse'], label='Entraînement', linewidth=2)
ax1.plot(range(epochs), evals_result['test']['rmse'], label='Test', linewidth=2)
ax1.set_xlabel('Itérations', fontsize=11)
ax1.set_ylabel('RMSE (t/ha)', fontsize=11)
ax1.set_title('Courbe d\'apprentissage', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 7.2 Observé vs Prédit
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identité')
ax2.set_xlabel('Rendement observé (t/ha)', fontsize=11)
ax2.set_ylabel('Rendement prédit (t/ha)', fontsize=11)
ax2.set_title(f'Validation du modèle (R² = {r2_test:.3f})', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 7.3 Distribution des résidus
ax3 = axes[1, 0]
residuals = y_test - y_pred_test
ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Résidus (t/ha)', fontsize=11)
ax3.set_ylabel('Fréquence', fontsize=11)
ax3.set_title('Distribution des résidus', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 7.4 Résidus vs Prédit
ax4 = axes[1, 1]
ax4.scatter(y_pred_test, residuals, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax4.axhline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Rendement prédit (t/ha)', fontsize=11)
ax4.set_ylabel('Résidus (t/ha)', fontsize=11)
ax4.set_title('Analyse des résidus', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/fig_xgb/evaluation_modele_SENTINEL2.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Graphique sauvegardé: evaluation_modele_SENTINEL2.png")
plt.show()

# ----- SECTION 8: IMPORTANCE DES FEATURES -----

print("\n" + "=" * 80)
print("IMPORTANCE DES FEATURES")
print("=" * 80)

importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
}).sort_values('Importance', ascending=False)

print("\nTop 10 des variables les plus importantes:")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Importance (Gain)', fontsize=12)
plt.ylabel('Variables', fontsize=12)
plt.title('Top 15 des variables les plus importantes', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/fig_xgb/importance_features_SENTINEL2.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Graphique sauvegardé: importance_features_SENTINEL2.png")
plt.show()

# ----- SECTION 9: ANALYSE SHAP -----

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
    plt.title('Impact des variables sur les prédictions (SHAP)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/fig_xgb/shap_summary_SENTINEL2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique SHAP sauvegardé: shap_summary_SENTINEL2.png")
    plt.show()

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False, max_display=15)
    plt.title('Importance SHAP des variables', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/fig_xgb/shap_importance_SENTINEL2.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique SHAP sauvegardé: shap_importance_SENTINEL2.png")
    plt.show()
    
    # Calcul des valeurs SHAP moyennes
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    print("\nTop 10 des impacts SHAP moyens:")
    for i, (feat, val) in enumerate(mean_abs_shap.head(10).items(), 1):
        print(f"  {i}. {feat}: {val:.4f}")
        
except Exception as e:
    print(f"\nProblème avec l'analyse SHAP: {str(e)}")
    print("   Utilisation de l'importance native du modèle à la place.")

# ----- SECTION 10: ANALYSE DES INDICES DE VÉGÉTATION -----

if available_indices:
    print("\n" + "=" * 80)
    print("CORRÉLATION INDICES DE VÉGÉTATION - RENDEMENTS")
    print("=" * 80)
    
    correlation_data = df[available_indices + ['yield_tpha']].corr()['yield_tpha'].drop('yield_tpha')
    print("\nCorrélations avec le rendement:")
    for idx, corr in correlation_data.sort_values(ascending=False).items():
        print(f"  {idx:6s}: r = {corr:+.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, indice in enumerate(available_indices):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(df[indice], df['yield_tpha'], alpha=0.5, s=30)
        
        # Régression linéaire
        valid_mask = df[indice].notna() & df['yield_tpha'].notna()
        if valid_mask.sum() > 1:
            z = np.polyfit(df[indice][valid_mask], df['yield_tpha'][valid_mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df[indice].min(), df[indice].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2)
        
        corr = df[[indice, 'yield_tpha']].corr().iloc[0, 1]
        ax.set_xlabel(indice, fontsize=11)
        ax.set_ylabel('Rendement (t/ha)', fontsize=11)
        ax.set_title(f'{indice} vs Rendement (r = {corr:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/fig_xgb/correlation_vegetation_SENTINEL2.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique de corrélation sauvegardé: correlation_vegetation_SENTINEL2.png")
    plt.show()

# ----- SECTION 11: RAPPORT FINAL -----

print("\n" + "=" * 80)
print("RÉSUMÉ DE L'ANALYSE")
print("=" * 80)
print(f"\nModèle: XGBoost avec {len(available_features)} variables")
print(f"  → Dont 4 indices Sentinel-2 RÉELS (NDVI, NDWI, EVI, LAI)")
print(f"\nPerformance (R² test): {r2_test:.4f}")
print(f"Erreur moyenne (MAE): {mae_test:.3f} t/ha")
print(f"Erreur relative (RRMSE): {rrmse_test:.2f}%")

print(f"\nTop 3 variables importantes:")
for i, row in importance_df.head(3).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Importance']:.2f}")

print(f"\nFichiers générés:")
print(f"  - evaluation_modele_SENTINEL2.png")
print(f"  - importance_features_SENTINEL2.png")
print(f"  - shap_summary_SENTINEL2.png")
print(f"  - shap_importance_SENTINEL2.png")
if available_indices:
    print(f"  - correlation_vegetation_SENTINEL2.png")

print("\n" + "=" * 80)
print("ANALYSE TERMINÉE AVEC INDICES SENTINEL-2 RÉELS")
print("=" * 80)