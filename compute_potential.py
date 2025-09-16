import glob
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy.ndimage import sobel
from tqdm import tqdm

tile_size = 1000
R_max = 1500

weights = {
    "solar": 0.5,
    "ndvi": 0.2,
    "slope": 0.2,
    "elevation": 0.1
}

# Load NDVI and cloud fraction
ndvi_median = np.load("ndvi_median.npy")
cloud_fraction = np.load("cloud_fraction.npy")

height, width = ndvi_median.shape

# Load DEM and resample
dem_file = "zone_dem.tif"
with rasterio.open(dem_file) as dem_src:
    dem = dem_src.read(
        1,
        out_shape=(height, width),
        resampling=Resampling.bilinear
    )

# Compute slope
dx = sobel(dem, axis=1)
dy = sobel(dem, axis=0)
slope = np.sqrt(dx**2 + dy**2)
slope_deg = np.degrees(np.arctan(slope))

slope_factor = np.clip(1 - slope_deg / 45, 0, 1)

# Normalized elevation
elevation_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
elevation_factor = 1 - elevation_norm  # high altitude penalty

# GeoTIFF output
safe_folder = sorted(glob.glob("downloaded data/*.SAFE"))[0]
red_path = glob.glob(os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA", "R10m", "*_B04_10m.jp2"))[0]

with rasterio.open(red_path) as src:
    profile = src.profile
    transform = src.transform

profile.update(
    dtype=rasterio.float32,
    count=1,
    compress='lzw',
    driver='GTiff',
    height=height,
    width=width,
    transform=transform
)

solar_score_array = np.zeros((height, width), dtype=np.float32)
output_file = "solar_score.tif"

with rasterio.open(output_file, "w", **profile) as dst:

    tiles_y = (height + tile_size - 1) // tile_size
    tiles_x = (width + tile_size - 1) // tile_size

    for ty in tqdm(range(tiles_y), desc="Scriem GeoTIFF tile-uri"):
        for tx in range(tiles_x):
            row_off = ty * tile_size
            col_off = tx * tile_size
            h = min(tile_size, height - row_off)
            w = min(tile_size, width - col_off)

            ndvi_tile = ndvi_median[row_off:row_off+h, col_off:col_off+w]
            cloud_tile = cloud_fraction[row_off:row_off+h, col_off:col_off+w]
            slope_tile = slope_factor[row_off:row_off+h, col_off:col_off+w]
            elevation_tile = elevation_factor[row_off:row_off+h, col_off:col_off+w]

            # Solar radiation factor
            solar_tile = R_max * (1 - 0.3 * np.clip(ndvi_tile, 0, 1)) * (1 - cloud_tile)
            solar_norm = (solar_tile - solar_tile.min()) / (solar_tile.max() - solar_tile.min() + 1e-6)

            # Vegetation penalty
            ndvi_factor = np.clip(1 - ndvi_tile / 0.45, 0, 1)

            # Final score
            score_tile = (
                weights["solar"] * solar_norm +
                weights["ndvi"] * ndvi_factor +
                weights["slope"] * slope_tile +
                weights["elevation"] * elevation_tile
            )

            solar_score_array[row_off:row_off+h, col_off:col_off+w] = score_tile
            dst.write(score_tile.astype(np.float32), 1, window=Window(col_off, row_off, w, h))

print(f"Solar score map saved as {output_file}")

np.save("solar_score.npy", solar_score_array)
print("Solar score saved also as 'solar_score.npy'")
