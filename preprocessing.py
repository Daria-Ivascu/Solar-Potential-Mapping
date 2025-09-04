import ee
import geemap

ee.Authenticate()
ee.Initialize()

aoi = ee.Geometry.Rectangle([72.5, 18.5, 74.0, 19.5])

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate('2018-01-01', '2023-12-31'))

def cloud_mask(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return cloud.rename('cloud')

cloud_stack = s2.map(cloud_mask)
cloud_freq = cloud_stack.mean().clip(aoi).rename('cloud_freq')

dem = ee.Image('COPERNICUS/DEM/GLO30').clip(aoi)
slope = ee.Terrain.slope(dem).rename('slope')
aspect = ee.Terrain.aspect(dem).rename('aspect')

result = cloud_freq.addBands([slope, aspect])

Map = geemap.Map()
Map.centerObject(aoi, 9)
Map.addLayer(cloud_freq, {'min':0, 'max':1, 'palette':['green', 'yellow', 'red']}, 'Cloud Freq')

geemap.ee_export_image(result, filename='gee_export.tif',
                       scale = 30, region = aoi, file_per_band = True)
