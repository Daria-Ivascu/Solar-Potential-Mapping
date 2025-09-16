import elevation
import rasterio
import numpy as np
from osgeo import gdal

bounds = (26, 44, 27, 45)

# Download DEM
dem_file = "zone_dem.tif"
elevation.clip(bounds=bounds, output=dem_file)

# Save as numpy
with rasterio.open(dem_file) as src:
    elevation_data = src.read(1).astype(np.float32)

np.save("elevation.npy", elevation_data)
print("Elevation saved as elevation.npy")

# Compute slope
slope_file = "slope.tif"
gdal.DEMProcessing(slope_file, dem_file, "slope", slopeFormat="degree")

with rasterio.open(slope_file) as src:
    slope_data = src.read(1).astype(np.float32)

np.save("slope.npy", slope_data)
print("Slope saved as slope.npy")
