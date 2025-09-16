import glob
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# -----------------------------
tile_size = 1000  # tile-uri pentru procesare
R_max = 1500
ndvi_weight = 0.3  # pentru factor NDVI în calculul solar
weights = {
    "solar": 0.7,
    "ndvi": 0.3
}

# -----------------------------
# Load NDVI median și cloud fraction
ndvi_median = np.load("ndvi_median.npy")
cloud_fraction = np.load("cloud_fraction.npy")

height, width = ndvi_median.shape

# -----------------------------
# Pregătim fișierul de output
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

# Array gol pentru scorul final
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

            # extragem NDVI și cloud fraction pentru tile
            ndvi_tile = ndvi_median[row_off:row_off+h, col_off:col_off+w]
            cloud_tile = cloud_fraction[row_off:row_off+h, col_off:col_off+w]

            # Calcul factor radiație solară (ajustat cu nori)
            solar_tile = R_max * (1 - ndvi_weight * np.clip(ndvi_tile, 0, 1)) * (1 - cloud_tile)
            solar_norm = (solar_tile - solar_tile.min()) / (solar_tile.max() - solar_tile.min() + 1e-6)

            # Factor NDVI invers (vegetation cover)
            ndvi_factor = np.clip(1 - ndvi_tile / 0.45, 0, 1)

            # Scor final ponderat
            score_tile = weights["solar"] * solar_norm + weights["ndvi"] * ndvi_factor

            # Scriem în array-ul complet
            solar_score_array[row_off:row_off+h, col_off:col_off+w] = score_tile

            # Scriem și în GeoTIFF
            dst.write(score_tile.astype(np.float32), 1, window=Window(col_off, row_off, w, h))

print(f"✅ Harta scor solar + vegetation salvată ca {output_file}")

# -----------------------------
# Salvăm scorul în format .npy pentru procesare ulterioară
np.save("solar_score.npy", solar_score_array)
print("✅ Scorul solar salvat și în 'solar_score.npy'")
