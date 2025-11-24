import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
print("="*70)
print("Modèle TabResNet pour la prédiction des rendements agricoles")
print("Intégration des indices de végétation (NDVI, NDWI, EVI, LAI)")
print("="*70)

# --- 1) Chargement des données ---
data_path = r'/home/snabraham6/#modele_deep_learning/data_model/data_final/fusion_complete_toutes_donnees.csv'
df = pd.read_csv(data_path)

print(f"\nDimensions des données: {df.shape}")
print(f"Colonnes disponibles: {df.columns.tolist()}")

# --- 2) Préparation de la variable cible ---
target = 'yield_tpha'
print(f"\n{'='*70}")
print(f"VARIABLE CIBLE: {target}")
print(f"{'='*70}")
print(df[target].describe())

# --- 3) INTÉGRATION DES INDICES SENTINEL-2 (RÉELS) ---
print(f"\n{'='*70}")
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
    df = df.merge(df_indices, on='Field', how='left')
    
    # Renommage
    df['NDVI'] = df['NDVI_moyen']
    df['NDWI'] = df['NDWI_moyen']
    df['EVI'] = df['EVI_moyen']
    df['LAI'] = df['LAI_moyen']
    
    print("\nIndices intégrés - Plages:")
    for idx in ['NDVI', 'NDWI', 'EVI', 'LAI']:
        valid = df[idx].dropna()
        if len(valid) > 0:
            print(f"  {idx}: [{valid.min():.3f}, {valid.max():.3f}], mean={valid.mean():.3f}")
    
except Exception as e:
    print(f"ERREUR: {e}")
    print("Utilisation de données simulées")
    np.random.seed(42)
    df['NDVI'] = np.random.uniform(0.3, 0.9, len(df))
    df['NDWI'] = np.random.uniform(-0.2, 0.3, len(df))
    df['EVI'] = np.random.uniform(0.2, 0.8, len(df))
    df['LAI'] = np.random.uniform(1.0, 6.0, len(df))
    print("\nIndices simulés générés")

################################# ENCODAGE DES VARIABLES CATÉGORIELLES ########################################
print(f"\n{'='*70}")
print("ENCODAGE DES VARIABLES CATÉGORIELLES")
print("="*70)

categorical_vars = ['drainage', 'station_name']
label_encoders = {}

for var in categorical_vars:
    if var in df.columns:
        le = LabelEncoder()
        df[f'{var}_encoded'] = le.fit_transform(df[var].astype(str))
        label_encoders[var] = le
        print(f"  - Encodage de '{var}' -> '{var}_encoded'")

################################## Ingénierie des features ###########################################
features_df = df.copy()

# Encodage zone et région
features_df['zone_cat'] = features_df['zone'].astype('category').cat.codes
features_df['region_cat'] = features_df['region'].astype('category').cat.codes

# Variables dérivées climatiques
if {'tmean', 'tmax', 'tmin', 'rain_mm', 'ppt_mm'}.issubset(features_df.columns):
    features_df['tmean2'] = features_df['tmean'] ** 2
    features_df['temp_range'] = features_df['tmax'] - features_df['tmin']
    features_df['rain_anomaly'] = features_df['rain_mm'] - features_df['rain_mm'].mean()
    features_df['ppt_ratio'] = features_df['rain_mm'] / (features_df['ppt_mm'] + 1e-6)
    print("\nVariables dérivées créées: tmean2, temp_range, rain_anomaly, ppt_ratio")

################################# Sélection des features #########################################
feature_candidates = [
    'region_cat', 'zone_cat', 'year',
    'g_pente_mo', 'tmean', 'tmax', 'tmin',
    'rain_mm', 'ppt_mm', 'snow_cm',
    'NDVI', 'NDWI', 'EVI', 'LAI',
    'drainage_encoded', 'station_name_encoded',
    'tmean2', 'temp_range', 'rain_anomaly', 'ppt_ratio'
]

available_features = [c for c in feature_candidates if c in features_df.columns]
missing_features = [c for c in feature_candidates if c not in features_df.columns]

if missing_features:
    print(f"\n⚠ Colonnes absentes ignorées : {missing_features}")

print(f"\n✓ Variables retenues : {len(available_features)} features")
print(f"  {available_features}")

######################################### Préparation X et y #######################################
X_raw = features_df[available_features].copy()
y = features_df[target].copy()

# Nettoyage
valid_idx = ~y.isna()
X_raw = X_raw[valid_idx]
y = y[valid_idx]
years = features_df.loc[valid_idx, 'year']

print(f"\nÉchantillons valides: {len(y)}")
print(f"Rendement moyen: {y.mean():.2f} t/ha (σ={y.std():.2f})")

# --- 8) Prétraitement ---
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

################################# Division temporelle des données (2010-2020 train, 2021-2023 test)#########################################
train_mask = years <= 2020
X_train_df = X_scaled[train_mask]
X_test_df = X_scaled[~train_mask]
y_train = y[train_mask]
y_test = y[~train_mask]

print(f"\n{'='*70}")
print("DIVISION TEMPORELLE DES DONNÉES")
print(f"{'='*70}")
print(f"Entraînement (2010-2020): {len(X_train_df)} échantillons ({len(X_train_df)/len(X_scaled)*100:.1f}%)")
print(f"Test (2021-2023): {len(X_test_df)} échantillons ({len(X_test_df)/len(X_scaled)*100:.1f}%)")

############################## Conversion en tenseurs PyTorch ###############################################
X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Split train → validation pour early stopping
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.15, random_state=42
)

############################# Définition du modèle TabResNet ################################################################
class TabResNetReg(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, n_blocks=1, p_drop=0.5):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.drop0 = nn.Dropout(p_drop)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p_drop)
            ) for _ in range(n_blocks)
        ])
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.drop0(x)
        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.relu(out + residual)
        return self.output_layer(x)

# --- 12) Entraînement du modèle ---
print(f"\n{'='*70}")
print("ENTRAÎNEMENT DU MODÈLE TABRESNET (ARCHITECTURE ULTRA-LÉGÈRE)")
print(f"{'='*70}")

input_dim = X_tr.shape[1]
output_dim = 1  # Prédiction d'une seule valeur (yield_tpha)

model = TabResNetReg(input_dim, output_dim, hidden_dim=16, n_blocks=1, p_drop=0.5)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # weight_decay augmenté
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15
)

print(f"\nArchitecture du modèle:")
print(f"  - Input dimension: {input_dim}")
print(f"  - Hidden dimension: 16 (réduit de 64)")
print(f"  - Nombre de blocs résiduels: 1 (réduit de 2)")
print(f"  - Dropout: 0.5 (augmenté de 0.3)")
print(f"  - Output dimension: {output_dim}")
print(f"\nNombre total de paramètres: {sum(p.numel() for p in model.parameters())}")
print(f"Ratio paramètres/observations: {sum(p.numel() for p in model.parameters()) / len(X_tr):.1f}:1")

best_val_loss = float('inf')
patience, wait = 30, 0
train_losses, val_losses = [], []

print("\nEntraînement en cours...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    preds_tr = model(X_tr)
    loss_tr = criterion(preds_tr, y_tr)
    loss_tr.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_val = model(X_val)
        loss_val = criterion(preds_val, y_val).item()

    train_losses.append(loss_tr.item())
    val_losses.append(loss_val)
    
    scheduler.step(loss_val)

    if loss_val < best_val_loss:
        best_val_loss = loss_val
        wait = 0
        torch.save(model.state_dict(), 'best_tabresnet_yield.pt')
    else:
        wait += 1
        if wait >= patience:
            print(f"  Early stopping à l'époque {epoch+1}")
            break

    if (epoch+1) % 50 == 0:
        print(f"  Epoch {epoch+1}: Train loss={loss_tr.item():.4f}, Val loss={loss_val:.4f}")

print(f"\n✓ Entraînement terminé")
print(f"  Meilleure perte de validation: {best_val_loss:.4f}")

# --- 13) Chargement du meilleur modèle et évaluation ---
model.load_state_dict(torch.load('best_tabresnet_yield.pt'))
model.eval()

with torch.no_grad():
    y_train_pred = model(X_train_tensor).numpy().flatten()
    y_test_pred = model(X_test_tensor).numpy().flatten()

# Métriques
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rrmse = (rmse / y_true.mean()) * 100
    return r2, rmse, mae, rrmse

r2_train, rmse_train, mae_train, rrmse_train = compute_metrics(y_train.values, y_train_pred)
r2_test, rmse_test, mae_test, rrmse_test = compute_metrics(y_test.values, y_test_pred)

print(f"\n{'='*70}")
print("RÉSULTATS DU MODÈLE")
print(f"{'='*70}")
print(f"\nEntraînement (2010-2020):")
print(f"  R² = {r2_train:.3f}")
print(f"  RMSE = {rmse_train:.3f} t/ha")
print(f"  MAE = {mae_train:.3f} t/ha")
print(f"  RRMSE = {rrmse_train:.2f}%")

print(f"\nTest (2021-2023):")
print(f"  R² = {r2_test:.3f}")
print(f"  RMSE = {rmse_test:.3f} t/ha")
print(f"  MAE = {mae_test:.3f} t/ha")
print(f"  RRMSE = {rrmse_test:.2f}%")

# --- 14) VISUALISATIONS ---
print(f"\n{'='*70}")
print("GÉNÉRATION DES VISUALISATIONS")
print(f"{'='*70}")

# Figure 1: Évaluation du modèle
fig1 = plt.figure(figsize=(16, 12))

# 1.1 Courbe d'apprentissage
ax1 = plt.subplot(3, 3, 1)
ax1.plot(train_losses, label='Train Loss', linewidth=1.5, alpha=0.8)
ax1.plot(val_losses, label='Validation Loss', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Époque', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax1.set_title('Courbe d\'apprentissage', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 1.2 Prédictions vs Observations - Train
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(y_train.values, y_train_pred, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ligne 1:1')
ax2.set_xlabel('Rendement observé (t/ha)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Rendement prédit (t/ha)', fontsize=11, fontweight='bold')
ax2.set_title(f'Entraînement (R²={r2_train:.3f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 1.3 Prédictions vs Observations - Test
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(y_test.values, y_test_pred, alpha=0.7, s=60, c='orange', edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ligne 1:1')
ax3.set_xlabel('Rendement observé (t/ha)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Rendement prédit (t/ha)', fontsize=11, fontweight='bold')
ax3.set_title(f'Test (R²={r2_test:.3f})', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 1.4 Distribution des résidus - Train
residuals_train = y_train.values - y_train_pred
ax4 = plt.subplot(3, 3, 4)
ax4.hist(residuals_train, bins=25, alpha=0.75, color='steelblue', edgecolor='black')
ax4.axvline(0, color='darkred', linestyle='--', linewidth=2)
ax4.axvline(residuals_train.mean(), color='blue', linestyle=':', linewidth=2)
ax4.set_xlabel('Résidus (t/ha)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
ax4.set_title(f'Distribution résidus Train (μ={residuals_train.mean():.2f})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 1.5 Distribution des résidus - Test
residuals_test = y_test.values - y_test_pred
ax5 = plt.subplot(3, 3, 5)
ax5.hist(residuals_test, bins=20, alpha=0.75, color='coral', edgecolor='black')
ax5.axvline(0, color='darkred', linestyle='--', linewidth=2)
ax5.axvline(residuals_test.mean(), color='blue', linestyle=':', linewidth=2)
ax5.set_xlabel('Résidus (t/ha)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
ax5.set_title(f'Distribution résidus Test (μ={residuals_test.mean():.2f})', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 1.6 Résidus vs Prédictions - Train
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(y_train_pred, residuals_train, alpha=0.6, s=50, c='steelblue')
ax6.axhline(0, color='darkred', linestyle='--', linewidth=2)
ax6.set_xlabel('Prédictions (t/ha)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Résidus (t/ha)', fontsize=11, fontweight='bold')
ax6.set_title('Résidus vs Prédictions - Train', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 1.7 Résidus vs Prédictions - Test
ax7 = plt.subplot(3, 3, 7)
ax7.scatter(y_test_pred, residuals_test, alpha=0.7, s=60, c='orange')
ax7.axhline(0, color='darkred', linestyle='--', linewidth=2)
ax7.set_xlabel('Prédictions (t/ha)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Résidus (t/ha)', fontsize=11, fontweight='bold')
ax7.set_title('Résidus vs Prédictions - Test', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 1.8 Q-Q Plot - Train
ax8 = plt.subplot(3, 3, 8)
stats.probplot(residuals_train, dist="norm", plot=ax8)
ax8.set_title('Q-Q Plot Train (normalité)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 1.9 Q-Q Plot - Test
ax9 = plt.subplot(3, 3, 9)
stats.probplot(residuals_test, dist="norm", plot=ax9)
ax9.set_title('Q-Q Plot Test (normalité)', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/resnet_model/figures_RB/evaluation_tabresnet.png', dpi=300, bbox_inches='tight')
print("  ✓ Graphiques d'évaluation sauvegardés")
plt.show()

# --- 15) CARTOGRAPHIE DES RENDEMENTS ---
print("\nGénération de la cartographie des rendements...")

# Prédictions sur l'ensemble complet
with torch.no_grad():
    X_all_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_all_pred = model(X_all_tensor).numpy().flatten()

# Construction du dataframe de cartographie
cartography_df = features_df.loc[X_scaled.index].copy()
cartography_df['yield_predicted'] = y_all_pred
cartography_df['yield_observed'] = y.loc[X_scaled.index].values
cartography_df['residual'] = cartography_df['yield_observed'] - cartography_df['yield_predicted']
cartography_df['abs_error'] = np.abs(cartography_df['residual'])

# Figure 2: Cartographie
fig2 = plt.figure(figsize=(18, 12))

# Heatmap prédictions par zone/année
ax1 = plt.subplot(2, 3, 1)
pivot_pred = cartography_df.pivot_table(
    values='yield_predicted',
    index='zone',
    columns='year',
    aggfunc='mean'
)
sns.heatmap(pivot_pred, annot=True, fmt='.1f', cmap='YlGn', ax=ax1,
            cbar_kws={'label': 'Rendement (t/ha)'}, linewidths=0.5)
ax1.set_title('Rendements prédits par zone et année', fontsize=13, fontweight='bold')
ax1.set_xlabel('Année', fontsize=11)
ax1.set_ylabel('Zone', fontsize=11)

# Heatmap observations
ax2 = plt.subplot(2, 3, 2)
pivot_obs = cartography_df.pivot_table(
    values='yield_observed',
    index='zone',
    columns='year',
    aggfunc='mean'
)
sns.heatmap(pivot_obs, annot=True, fmt='.1f', cmap='YlGn', ax=ax2,
            cbar_kws={'label': 'Rendement (t/ha)'}, linewidths=0.5)
ax2.set_title('Rendements observés par zone et année', fontsize=13, fontweight='bold')
ax2.set_xlabel('Année', fontsize=11)
ax2.set_ylabel('Zone', fontsize=11)

# Heatmap résidus
ax3 = plt.subplot(2, 3, 3)
pivot_resid = cartography_df.pivot_table(
    values='residual',
    index='zone',
    columns='year',
    aggfunc='mean'
)
sns.heatmap(pivot_resid, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax3,
            cbar_kws={'label': 'Résidu (t/ha)'}, linewidths=0.5)
ax3.set_title('Résidus par zone et année', fontsize=13, fontweight='bold')
ax3.set_xlabel('Année', fontsize=11)
ax3.set_ylabel('Zone', fontsize=11)

# Rendements par région
ax4 = plt.subplot(2, 3, 4)
region_stats = cartography_df.groupby('region').agg({
    'yield_observed': 'mean',
    'yield_predicted': 'mean'
}).reset_index()

x_pos = np.arange(len(region_stats))
width = 0.35
ax4.bar(x_pos - width/2, region_stats['yield_observed'], width,
        label='Observé', alpha=0.85, color='forestgreen', edgecolor='black')
ax4.bar(x_pos + width/2, region_stats['yield_predicted'], width,
        label='Prédit', alpha=0.85, color='royalblue', edgecolor='black')
ax4.set_xlabel('Région', fontsize=11, fontweight='bold')
ax4.set_ylabel('Rendement moyen (t/ha)', fontsize=11, fontweight='bold')
ax4.set_title('Rendements moyens par région', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'Rég {int(r)}' for r in region_stats['region']], fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Évolution temporelle
ax5 = plt.subplot(2, 3, 5)
temporal = cartography_df.groupby('year').agg({
    'yield_observed': ['mean', 'std'],
    'yield_predicted': ['mean', 'std']
})

years_plot = temporal.index
ax5.plot(years_plot, temporal[('yield_observed', 'mean')], 'o-', linewidth=2.5,
         markersize=8, label='Observé', color='darkgreen')
ax5.fill_between(years_plot,
                  temporal[('yield_observed', 'mean')] - temporal[('yield_observed', 'std')],
                  temporal[('yield_observed', 'mean')] + temporal[('yield_observed', 'std')],
                  alpha=0.2, color='darkgreen')

ax5.plot(years_plot, temporal[('yield_predicted', 'mean')], 's-', linewidth=2.5,
         markersize=8, label='Prédit', color='navy')
ax5.fill_between(years_plot,
                  temporal[('yield_predicted', 'mean')] - temporal[('yield_predicted', 'std')],
                  temporal[('yield_predicted', 'mean')] + temporal[('yield_predicted', 'std')],
                  alpha=0.2, color='navy')

ax5.set_xlabel('Année', fontsize=11, fontweight='bold')
ax5.set_ylabel('Rendement (t/ha)', fontsize=11, fontweight='bold')
ax5.set_title('Évolution temporelle des rendements', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# Précision spatiale par zone
ax6 = plt.subplot(2, 3, 6)
spatial_error = cartography_df.groupby('zone').agg({
    'abs_error': 'mean',
    'yield_observed': 'mean'
}).reset_index()
spatial_error['rrmse'] = (spatial_error['abs_error'] / spatial_error['yield_observed']) * 100

colors6 = plt.cm.RdYlGn_r(spatial_error['rrmse'] / spatial_error['rrmse'].max())
ax6.barh(spatial_error['zone'].astype(str), spatial_error['rrmse'],
         color=colors6, alpha=0.8, edgecolor='black')
ax6.set_xlabel('RRMSE (%)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Zone', fontsize=11, fontweight='bold')
ax6.set_title('Précision du modèle par zone', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/resnet_model/figures_RB/cartographie_tabresnet.png', dpi=300, bbox_inches='tight')
print("  ✓ Cartographie sauvegardée")
plt.show()

# --- 16) Analyse des indices de végétation ---
fig3 = plt.figure(figsize=(16, 10))

for i, idx_name in enumerate(['NDVI', 'NDWI', 'EVI', 'LAI'], 1):
    ax = plt.subplot(2, 2, i)
    
    scatter = ax.scatter(cartography_df[idx_name],
                         cartography_df['yield_observed'],
                         c=cartography_df['year'],
                         cmap='viridis',
                         alpha=0.6,
                         s=50,
                         edgecolors='black',
                         linewidth=0.3)
    
    z = np.polyfit(cartography_df[idx_name], cartography_df['yield_observed'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(cartography_df[idx_name].min(), cartography_df[idx_name].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label='Tendance')
    
    correlation = np.corrcoef(cartography_df[idx_name], cartography_df['yield_observed'])[0, 1]
    
    ax.set_xlabel(idx_name, fontsize=11, fontweight='bold')
    ax.set_ylabel('Rendement observé (t/ha)', fontsize=11, fontweight='bold')
    ax.set_title(f'{idx_name} vs Rendement (r={correlation:.3f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if i == 1:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Année', fontsize=10)

plt.tight_layout()
plt.savefig(r'/home/snabraham6/#modele_deep_learning/resnet_model/figures/indices_vegetation_tabresnet.png', dpi=300, bbox_inches='tight')
print("  ✓ Analyse des indices de végétation sauvegardée")
plt.close()

# --- 17) Export des résultats ---
results_export = cartography_df[['region', 'zone', 'year', 'Field',
                                  'yield_observed', 'yield_predicted', 'residual', 'abs_error',
                                  'NDVI', 'NDWI', 'EVI', 'LAI',
                                  'tmean', 'tmax', 'tmin', 'rain_mm']].copy()

results_export['model_used'] = 'TabResNet'
results_export.to_csv(r'/home/snabraham6/#modele_deep_learning/resnet_model/data/predictions_tabresnet.csv', index=False)
print("\n  ✓ Résultats détaillés exportés")
