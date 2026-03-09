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
# Structure préparée pour NDVI, EVI, LAI
# Si ces colonnes existent dans vos données, elles seront automatiquement intégrées
# Sinon, vous pouvez les ajouter via fusion avec un autre fichier

vegetation_indices = ['NDVI', 'EVI', 'LAI']
available_indices = [col for col in vegetation_indices if col in df.columns]

if available_indices:
    print(f"\nIndices de végétation détectés: {', '.join(available_indices)}")
else:
    print("\nAucun indice de végétation détecté dans les données actuelles.")
    print("Pour les intégrer, ajoutez les colonnes NDVI, EVI, LAI au CSV.")
    
    # Génération de données synthétiques pour démonstration (à remplacer par vos vraies données)
    print("\nGénération de données synthétiques pour démonstration...")
    np.random.seed(42)
    df['NDVI'] = np.random.uniform(0.3, 0.9, len(df))
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
    'tmean', 'tmax', 'tmin', 'rain_mm', 'ppt_mm',
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
    'learning_rate': 0.05,        
    'max_depth': 5,               
    'subsample': 0.75,            
    'colsample_bytree': 0.75,     
    'min_child_weight': 5,        
    'gamma': 0.2,                 
    'reg_alpha': 0.3,             
    'reg_lambda': 1.5,            
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
    num_boost_round=300,          
    evals=evals,
    early_stopping_rounds=30,     
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
plt.title(f'Test du modèle (R² = {r2_test:.3f}), (RMSE = {rmse_test:.3f}), (MAE = {mae_test:.3f})', fontsize=10, fontweight='bold')
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

# Marquer les données de test dans le dataframe
df['is_test'] = False
df.loc[X_test.index, 'is_test'] = True

# Sauvegarde des résultats
output_df = df[[
    'Field', 'year', 'station_name', 'drainage',
    'yield_tpha', 'yield_predicted', 'residual', 'abs_error'
] + available_indices].copy()

output_df.to_csv(r'/home/snabraham6/#modele_deep_learning/xgboost_model/data/predictions_rendements.csv', index=False)
print(f"\nRésultats sauvegardés: predictions_rendements.csv")

############################ GRAPHIQUE PAR CHAMP F1-F10 ###################################
print("\n--- Génération des graphiques par champ ---")

# Récupérer la liste unique des champs et filtrer uniquement F1 à F10
all_fields = sorted(df['Field'].unique())
unique_fields = [f for f in all_fields if f.startswith('F') and f[1:].isdigit() and 1 <= int(f[1:]) <= 10]
unique_fields = sorted(unique_fields, key=lambda x: int(x[1:]))  # Trier par numéro

n_fields = len(unique_fields)

print(f"Nombre de champs à afficher: {n_fields}")
print(f"Champs sélectionnés: {unique_fields}")

# Calculer la disposition de la grille (3 colonnes)
n_cols = 3
n_rows = (n_fields + n_cols - 1) // n_cols

print(f"Grille: {n_rows} lignes x {n_cols} colonnes")

# Créer la figure avec subplots
fig = plt.figure(figsize=(18, 6 * n_rows))
gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

# Graphiques individuels par champ
for idx, field in enumerate(unique_fields):
    # Calculer la position dans la grille
    row = idx // n_cols
    col = idx % n_cols
    
    print(f"Champ {field}: position [{row}, {col}]")
    
    ax = fig.add_subplot(gs[row, col])
    
    # Filtrer les données de test pour ce champ
    field_test_data = df[(df['Field'] == field) & (df['is_test'] == True)].copy()
    
    if len(field_test_data) > 0:
        y_field_test = field_test_data['yield_tpha'].values
        y_field_pred = field_test_data['yield_predicted'].values
        
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

plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/test_par_champ.png', 
            dpi=300, bbox_inches='tight')
print("Graphique sauvegardé: test_par_champ.png")
plt.show()

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
    
############################ ANALYSE TEMPORALE PAR CHAMP ###################################

# Filtrer et trier les champs pour exclure les IRDA
fields_to_plot = [field for field in df['Field'].unique() if not field.startswith('IRDA')]

# Fonction de tri personnalisée pour extraire le numéro du champ
def extract_field_number(field_name):
    """Extrait le numéro du champ (ex: 'F10' -> 10)"""
    if field_name.startswith('F'):
        try:
            return int(field_name[1:])
        except:
            return 999
    return 999

# Trier les champs par numéro
fields_to_plot = sorted(fields_to_plot, key=extract_field_number)
print(f"Champs à tracer (triés): {fields_to_plot}")

# PALETTE DE COULEURS AMÉLIORÉE - Plus claire et distincte
custom_colors = [
    '#2E86AB',  # Bleu vif
    '#A23B72',  # Magenta
    '#F18F01',  # Orange
    '#C73E1D',  # Rouge-orange
    '#6A994E',  # Vert
    '#BC4B51',  # Rouge rosé
    '#8D5B4C',  # Brun
    '#5C7F67',  # Vert sauge
    '#D4A574',  # Beige doré
    '#7A6F9B',  # Violet
    '#E63946',  # Rouge vif
    '#06AED5',  # Cyan
    '#DD6E42',  # Terracotta
    '#4EA8DE',  # Bleu clair
    '#99C24D',  # Vert lime
    '#B07BAC',  # Mauve
    '#F4A259',  # Orange pêche
    '#5E8C61',  # Vert foncé
    '#E5989B',  # Rose
    '#118AB2'   # Bleu-vert
]

n_fields = len(fields_to_plot)

# Étendre la palette si nécessaire
while len(custom_colors) < n_fields:
    custom_colors.extend(custom_colors[:min(20, n_fields - len(custom_colors))])

colors = custom_colors[:n_fields]

# FIGURE A : Graphiques séparés par champ (grille de subplots)
# Calculer la disposition optimale de la grille
n_cols = 3  # 3 colonnes pour une meilleure lisibilité
n_rows = int(np.ceil(n_fields / n_cols))

fig_a, axes_a = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
fig_a.suptitle('Figure a : Évolution temporelle des rendements par champ', 
               fontsize=14, fontweight='bold', y=0.995)

# Aplatir le tableau d'axes pour faciliter l'itération
axes_a_flat = axes_a.flatten() if n_fields > 1 else [axes_a]

# Tracer chaque champ dans son propre subplot
for idx, field in enumerate(fields_to_plot):
    ax = axes_a_flat[idx]
    field_data = df[df['Field'] == field].sort_values('year')
    color = colors[idx]
    
    # Tracer observations avec ligne solide
    ax.plot(field_data['year'], field_data['yield_tpha'], 
            marker='o', color=color, linewidth=2.5, markersize=8,
            label='Observé', alpha=0.95)
    
    # Tracer prédictions avec ligne pointillée de la même couleur
    ax.plot(field_data['year'], field_data['yield_predicted'], 
            marker='s', linestyle='--', color=color, linewidth=2, markersize=6,
            label='Prédit', alpha=0.75)
    
    ax.set_title(f'{field}', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Année', fontsize=10)
    ax.set_ylabel('Rendement (t/ha)', fontsize=10)
    ax.legend(loc='best', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')

# Masquer les axes inutilisés
for idx in range(n_fields, len(axes_a_flat)):
    axes_a_flat[idx].axis('off')

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/analyse_temporelle_fig_a.png', 
            dpi=300, bbox_inches='tight')
print(f"Figure  sauvegardée: analyse_temporelle_fig_a.png")
plt.show()

# FIGURE B : Boxplot des erreurs par champ (sans IRDA) - ordonnés
fig_b, ax_b = plt.subplots(1, 1, figsize=(14, 6))

field_errors = [df[df['Field'] == field]['abs_error'].values 
                for field in fields_to_plot]

# Créer le boxplot avec les mêmes couleurs
bp = ax_b.boxplot(field_errors, labels=fields_to_plot, patch_artist=True, widths=0.6)

# Colorer chaque boîte avec la couleur correspondante du champ
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
    patch.set_linewidth(1.5)

# Colorer les autres éléments avec un style plus moderne
for element in ['whiskers', 'fliers', 'medians', 'caps']:
    for i, item in enumerate(bp[element]):
        if element == 'medians':
            item.set_color('#D62828')  # Rouge vif pour la médiane
            item.set_linewidth(2.5)
        else:
            # Associer la couleur du champ correspondant
            field_idx = i // 2 if element in ['whiskers', 'caps'] else i
            if field_idx < len(colors):
                if element == 'fliers':
                    item.set_markeredgecolor(colors[field_idx])
                    item.set_markerfacecolor('white')
                    item.set_markersize(5)
                    item.set_alpha(0.6)
                else:
                    item.set_color(colors[field_idx])
                    item.set_linewidth(1.5)

ax_b.set_xlabel('Champ', fontsize=12, fontweight='bold')
ax_b.set_ylabel('Erreur absolue (t/ha)', fontsize=12, fontweight='bold')
ax_b.set_title('Figure b : Distribution des erreurs par champ', 
               fontsize=14, fontweight='bold', pad=15)
ax_b.grid(True, alpha=0.3, axis='y', linestyle='--')
ax_b.set_facecolor('#FAFAFA')
plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/xgboost_model/figures/analyse_temporelle_fig_b.png', 
            dpi=300, bbox_inches='tight')
print(f"Figure b sauvegardée: analyse_temporelle_fig_b.png")
plt.show()
