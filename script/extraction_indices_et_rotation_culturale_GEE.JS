/////////////////////////////// Extraction des indices de végétation Sentinel-2 //////////////////////////////////////////

/////////////////////////// 1. CONFIGURATION - Modifier des paramètres /////////////////////////////////////////////////

var parcelles = ee.FeatureCollection('users/snabraham6/ME_Corn_Fields_270525');

// Définir la période d'étude (saison de croissance)
var dateDebut = '2010-05-01';  // Début mai
var dateFin = '2023-09-30';    // Fin septembre

// Seuil de couverture nuageuse maximum (%)
var seuilNuages = 20;

// Filtrer les parcelles (exclure IRDA)
// Exclure toutes les parcelles dont le Field contient "IRDA"
var parcellesFiltrees = parcelles.filter(
  ee.Filter.stringContains('Field', 'IRDA').not()
);


////////////////////////////////// 1B. CHARGEMENT AAFC CROP ROTATION /////////////////////////////////////////////////

print('Chargement de AAFC Annual Crop Inventory...');

var aafc = ee.ImageCollection('AAFC/ACI')
  .filterBounds(parcelles)
  .filterDate('2010-01-01', '2023-12-31');

print('Nombre d\'années AAFC disponibles:', aafc.size());

// Fonction pour extraire le type de culture par parcelle pour chaque année
var extraireCulture = function(image) {
  var annee = ee.Date(image.get('system:time_start')).get('year');
  
  var extractionCulture = function(parcelle) {
    var stats = image.reduceRegion({
      reducer: ee.Reducer.mode(), // Culture dominante dans la parcelle
      geometry: parcelle.geometry(),
      scale: 30, // Résolution AAFC
      maxPixels: 1e9
    });
    
    return parcelle.set({
      'crop_type': stats.get('landcover'),
      'year_aafc': annee
    });
  };
  
  return parcelles.map(extractionCulture);
};

// Extraire les cultures pour toutes les années
var rotationCultures = aafc.map(function(img) {
  return extraireCulture(img);
}).flatten();

// Visualiser la dernière année disponible (2023 ou la plus récente)
var derniereImageAAFC = aafc.sort('system:time_start', false).first();
Map.addLayer(derniereImageAAFC, {
  min: 1,
  max: 255,
  palette: ['yellow', 'green', 'brown', 'blue', 'orange']
}, 'AAFC Crop Types (dernière année)', false);


////////////////////////////////// 2. CHARGEMENT DES IMAGES SENTINEL-2 /////////////////////////////////////////////////

print('Chargement de la collection Sentinel-2...');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(parcellesFiltrees)
  .filterDate(dateDebut, dateFin)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', seuilNuages));

print('Nombre d\'images disponibles:', s2.size());


////////////////////////////////// 3. FONCTIONS DE CALCUL DES INDICES ////////////////////////////////////////////////////

function ajouterIndices(image) {
  // NDVI - Normalized Difference Vegetation Index
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // EVI - Enhanced Vegetation Index
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }).rename('EVI');
  
  // LAI approximatif (basé sur relation empirique avec NDVI)
  // LAI = 3.618 × NDVI - 0.118
  var lai = ndvi.multiply(3.618).subtract(0.118).clamp(0, 8).rename('LAI');
  
  // Ajouter la date comme propriété
  return image.addBands([ndvi, evi, lai])
              .set('date', image.date().format('YYYY-MM-dd'));
}

// Appliquer le calcul des indices à toutes les images
var avecIndices = s2.map(ajouterIndices);


/////////////////////////////////////// 4. EXTRACTION DES VALEURS PAR PARCELLE //////////////////////////////////////////

/// Moyenne de tous les indices sur la période complète
var indicesMoyens = avecIndices.select(['NDVI', 'EVI', 'LAI']).mean();

var extraireValeursMoyennes = function(parcelle) {
  var stats = indicesMoyens.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: parcelle.geometry(),
    scale: 10,  // Résolution de 10m
    maxPixels: 1e9
  });
  
/// Ajouter les statistiques à la parcelle
  return parcelle.set({
    'NDVI_moyen': stats.get('NDVI'),
    'EVI_moyen': stats.get('EVI'),
    'LAI_moyen': stats.get('LAI')
  });
};

var resultatMoyen = parcellesFiltrees.map(extraireValeursMoyennes);

// Méthode B : Valeurs maximales (pic de végétation)
var indicesMax = avecIndices.select(['NDVI', 'EVI', 'LAI']).max();

var extraireValeursMax = function(parcelle) {
  var stats = indicesMax.reduceRegion({
    reducer: ee.Reducer.max(),
    geometry: parcelle.geometry(),
    scale: 10,
    maxPixels: 1e9
  });
  
  return parcelle.set({
    'NDVI_max': stats.get('NDVI'),
    'EVI_max': stats.get('EVI'),
    'LAI_max': stats.get('LAI')
  });
};

var resultatMax = parcellesFiltrees.map(extraireValeursMax);

// Fusionner les deux résultats
var resultatFinal = resultatMoyen.map(function(f) {
  var maxFeature = resultatMax.filter(ee.Filter.eq('Field', f.get('Field'))).first();
  return f.copyProperties(maxFeature, ['NDVI_max', 'EVI_max', 'LAI_max']);
});


///////////////////////////////////////////////// 5. EXPORT DES RÉSULTATS ///////////////////////////////////////

// Export des indices de végétation
Export.table.toDrive({
  collection: resultatFinal,
  description: 'indices_vegetation_sentinel2',
  folder: 'EarthEngine_Exports',
  fileFormat: 'CSV',
  selectors: ['Field', 'year', 'NDVI_moyen', 'EVI_moyen', 'LAI_moyen',
              'NDVI_max', 'EVI_max', 'LAI_max']
});

// Export de la rotation des cultures AAFC
Export.table.toDrive({
  collection: rotationCultures,
  description: 'rotation_cultures_aafc_2010_2023',
  folder: 'EarthEngine_Exports',
  fileFormat: 'CSV',
  selectors: ['Field', 'year', 'crop_type', 'year_aafc']
});

print('Exports configurés. Cliquer sur "Run" dans le panneau Tasks pour lancer.');


//////////////////////////////////////////// 6. VISUALISATION //////////////////////////////////////////////////////

// Paramètres de visualisation pour NDVI
var visParamsNDVI = {
  bands: ['NDVI'],
  min: 0,
  max: 0.9,
  palette: ['brown', 'yellow', 'green', 'darkgreen']
};

// Paramètres pour LAI
var visParamsLAI = {
  bands: ['LAI'],
  min: 0,
  max: 6,
  palette: ['white', 'lightgreen', 'green', 'darkgreen']
};

// Centrer la carte sur les parcelles
Map.centerObject(parcellesFiltrees, 12);

// Ajouter les couches à la carte
Map.addLayer(indicesMoyens, visParamsNDVI, 'NDVI moyen');
Map.addLayer(indicesMoyens, visParamsLAI, 'LAI moyen', false);
Map.addLayer(parcellesFiltrees, {color: 'red', width: 2}, 'Parcelles');

// Afficher des statistiques
print('Résultats (5 premières parcelles):', resultatFinal.limit(5));
print('Nombre de parcelles traitées:', resultatFinal.size());
print('Rotation cultures (5 premiers enregistrements):', rotationCultures.limit(5));


//////////////////////////////////////// 7. ANALYSE TEMPORELLE - GRAPHIQUE NDVI FILTRÉ ///////////////////////////////////////////////////

// Graphique NDVI avec parcelles filtrées (exclut IRDA si configuré)
var chartNDVI = ui.Chart.image.seriesByRegion({
  imageCollection: avecIndices.select('NDVI'),
  regions: parcellesFiltrees, // Utilise les parcelles filtrées
  reducer: ee.Reducer.mean(),
  scale: 10,
  xProperty: 'system:time_start',
  seriesProperty: 'Field'
}).setOptions({
  title: 'Évolution du NDVI par parcelle (parcelles filtrées)',
  vAxis: {title: 'NDVI'},
  hAxis: {title: 'Date'},
  lineWidth: 2,
  pointSize: 3
});

print(chartNDVI);

// Graphique optionnel : Diversité des cultures par année
var annees = ee.List.sequence(2010, 2023);
var compteCultures = annees.map(function(annee) {
  var imgAnnee = aafc.filter(ee.Filter.calendarRange(annee, annee, 'year')).first();
  if (imgAnnee) {
    var culturesAnnee = imgAnnee.reduceRegions({
      collection: parcellesFiltrees,
      reducer: ee.Reducer.mode(),
      scale: 30
    });
    var unique = culturesAnnee.aggregate_count_distinct('mode');
    return ee.Feature(null, {'year': annee, 'nb_cultures': unique});
  }
  return ee.Feature(null, {'year': annee, 'nb_cultures': 0});
});

var chartRotation = ui.Chart.feature.byFeature({
  features: ee.FeatureCollection(compteCultures),
  xProperty: 'year',
  yProperties: ['nb_cultures']
}).setOptions({
  title: 'Diversité des cultures par année (Montérégie)',
  vAxis: {title: 'Nombre de types de cultures'},
  hAxis: {title: 'Année'},
  lineWidth: 2,
  pointSize: 4
});

print(chartRotation);
