/////////////////////////////// Extraction des indices de végétation Sentinel-2 //////////////////////////////////////////

/////////////////////////// 1. CONFIGURATION - Modifier des paramètres /////////////////////////////////////////////////

var parcelles = ee.FeatureCollection('users/snabraham6/ME_Corn_Fields_270525');

// Définir la période d'étude (saison de croissance)
var dateDebut = '2010-05-01';  // Début mai
var dateFin = '2023-09-30';    // Fin septembre

// Seuil de couverture nuageuse maximum (%)
var seuilNuages = 20;

////////////////////////////////// 2. CHARGEMENT DES IMAGES SENTINEL-2 /////////////////////////////////////////////////
// 

print('Chargement de la collection Sentinel-2...');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(parcelles)
  .filterDate(dateDebut, dateFin)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', seuilNuages));

print('Nombre d\'images disponibles:', s2.size());

////////////////////////////////////////////// Partie 4 : LANDSAT 5/7/8 (2010-2014) ///////////////////////////////////////////
// 

function maskLandsatClouds(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask);
}

function addLandsatIndices(image) {
  var b = image.select('SR_B2');
  var g = image.select('SR_B3');
  var r = image.select('SR_B4');
  var n = image.select('SR_B5');
  var s = image.select('SR_B6');
  
  var ndvi = n.subtract(r).divide(n.add(r)).rename('NDVI');
  var evi = n.subtract(r)
              .divide(n.add(r.multiply(6)).subtract(b.multiply(7.5)).add(1))
              .multiply(2.5).rename('EVI');
  var ndwi = g.subtract(n).divide(g.add(n)).rename('NDWI');
  var lai = evi.multiply(3.618).subtract(0.118).rename('LAI');
  
  return image.addBands([ndvi, evi, ndwi, gndvi, savi, lai])
              .select(['NDVI','EVI','NDWI','LAI']);
}

function getLandsatCollection(start, end) {
  var col5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    .filterBounds(ZONE_ETUDE.geometry())
    .filterDate(start, end)
    .map(maskLandsatClouds)
    .map(addLandsatIndices);
    
  var col7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterBounds(ZONE_ETUDE.geometry())
    .filterDate(start, end)
    .map(maskLandsatClouds)
    .map(addLandsatIndices);
    
  var col8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(ZONE_ETUDE.geometry())
    .filterDate(start, end)
    .map(maskLandsatClouds)
    .map(addLandsatIndices);
    
  return col5.merge(col7).merge(col8);
}

////////////////////////////////// 3. FONCTIONS DE CALCUL DES INDICES ////////////////////////////////////////////////////
// 

function ajouterIndices(image) {
  // NDVI - Normalized Difference Vegetation Index
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // NDWI - Normalized Difference Water Index
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  
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
  return image.addBands([ndvi, ndwi, evi, lai])
              .set('date', image.date().format('YYYY-MM-dd'));
}

// Appliquer le calcul des indices à toutes les images
var avecIndices = s2.map(ajouterIndices);


/////////////////////////////////////// 4. EXTRACTION DES VALEURS PAR PARCELLE //////////////////////////////////////////
//

/// Moyenne de tous les indices sur la période complète
var indicesMoyens = avecIndices.select(['NDVI', 'NDWI', 'EVI', 'LAI']).mean();

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
    'NDWI_moyen': stats.get('NDWI'),
    'EVI_moyen': stats.get('EVI'),
    'LAI_moyen': stats.get('LAI')
  });
};

var resultatMoyen = parcelles.map(extraireValeursMoyennes);

// Méthode B : Valeurs maximales (pic de végétation)
var indicesMax = avecIndices.select(['NDVI', 'NDWI', 'EVI', 'LAI']).max();

var extraireValeursMax = function(parcelle) {
  var stats = indicesMax.reduceRegion({
    reducer: ee.Reducer.max(),
    geometry: parcelle.geometry(),
    scale: 10,
    maxPixels: 1e9
  });
  
  return parcelle.set({
    'NDVI_max': stats.get('NDVI'),
    'NDWI_max': stats.get('NDWI'),
    'EVI_max': stats.get('EVI'),
    'LAI_max': stats.get('LAI')
  });
};

var resultatMax = parcelles.map(extraireValeursMax);

// Fusionner les deux résultats
var resultatFinal = resultatMoyen.map(function(f) {
  var maxFeature = resultatMax.filter(ee.Filter.eq('Field', f.get('Field'))).first();
  return f.copyProperties(maxFeature, ['NDVI_max', 'NDWI_max', 'EVI_max', 'LAI_max']);
});

///////////////////////////////////////////////// 5. EXPORT DES RÉSULTATS ///////////////////////////////////////
// 

// Export vers Google Drive (fichier CSV)
Export.table.toDrive({
  collection: resultatFinal,
  description: 'indices_vegetation_sentinel2',
  folder: 'EarthEngine_Exports',
  fileFormat: 'CSV',
  selectors: ['Field', 'year', 'NDVI_moyen', 'NDWI_moyen', 'EVI_moyen', 'LAI_moyen',
              'NDVI_max', 'NDWI_max', 'EVI_max', 'LAI_max']
});

print('Export configuré. Cliquer sur "Run" dans le panneau Tasks pour lancer.');

//////////////////////////////////////////// 6. VISUALISATION //////////////////////////////////////////////////////
// 

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
Map.centerObject(parcelles, 12);

// Ajouter les couches à la carte
Map.addLayer(indicesMoyens, visParamsNDVI, 'NDVI moyen');
Map.addLayer(indicesMoyens, visParamsLAI, 'LAI moyen', false);
Map.addLayer(parcelles, {color: 'red', width: 2}, 'Parcelles');

// Afficher des statistiques
print('Résultats (5 premières parcelles):', resultatFinal.limit(5));
print('Nombre de parcelles traitées:', resultatFinal.size());


//////////////////////////////////////// 7. ANALYSE TEMPORELLE ///////////////////////////////////////////////////
// 

// Créer un graphique de l'évolution du NDVI moyen sur la saison
var chartNDVI = ui.Chart.image.seriesByRegion({
  imageCollection: avecIndices.select('NDVI'),
  regions: parcelles,
  reducer: ee.Reducer.mean(),
  scale: 10,
  xProperty: 'system:time_start',
  seriesProperty: 'Field'
}).setOptions({
  title: 'Évolution du NDVI par parcelle',
  vAxis: {title: 'NDVI'},
  hAxis: {title: 'Date'},
  lineWidth: 2,
  pointSize: 3
});

print(chartNDVI);
