import ee
import geemap

ee.Authenticate()
ee.Initialize(project = 'solar-potential-mapping')

aoi = ee.Geometry.Polygon([
    [[20.0, 43.5], [29.7, 43.5], [29.7, 48.3], [20.0, 48.3]]
    ])

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(aoi)
      .filterDate('2023-01-01', '2023-12-31'))

def cloud_mask(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return cloud.rename('cloud')

cloud_stack = s2.map(cloud_mask)
cloud_freq = cloud_stack.mean().clip(aoi).rename('cloud_freq')

dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic().clip(aoi)
slope = ee.Terrain.slope(dem).rename('slope')
aspect = ee.Terrain.aspect(dem).rename('aspect')

result = cloud_freq.addBands([slope, aspect])

Map = geemap.Map()
Map.centerObject(aoi, 9)
Map.addLayer(cloud_freq, {'min':0, 'max':1, 'palette':['green', 'yellow', 'red']}, 'Cloud Freq')

# small_aoi = ee.Geometry.Rectangle([73, 18.9, 73.5, 19.1])
# geemap.ee_export_image(result, filename='gee_export.tif',
#                        scale = 30, region = aoi, file_per_band = True)

task = ee.batch.Export.image.toDrive(
    image = result,
    description = 'solar_potential',
    folder = 'GEE_exports',
    fileNamePrefix = 'gee_export',
    region = aoi,
    scale = 30,
    fileFormat = 'GeoTIFF'
)
task.start()
