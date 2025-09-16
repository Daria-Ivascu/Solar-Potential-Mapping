import glob
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import rasterio

# -----------------------------
tile_size = 1000  # dimensiunea fiecărui tile
sample_per_tile = 5000  # pixeli random per tile pentru antrenament

# -----------------------------
# Load NDVI, cloud fraction și scor ponderat
ndvi = np.load("ndvi_median.npy")
cloud_fraction = np.load("cloud_fraction.npy")
solar_score = np.load("solar_score.npy")

height, width = ndvi.shape

# -----------------------------
# Construim dataset ML prin sampling pe tile-uri
X_list = []
y_list = []

tiles_y = (height + tile_size - 1) // tile_size
tiles_x = (width + tile_size - 1) // tile_size

for ty in tqdm(range(tiles_y), desc="Sampling pixels for ML"):
    for tx in range(tiles_x):
        row_off = ty * tile_size
        col_off = tx * tile_size
        h = min(tile_size, height - row_off)
        w = min(tile_size, width - col_off)

        ndvi_tile = ndvi[row_off:row_off+h, col_off:col_off+w]
        cloud_tile = cloud_fraction[row_off:row_off+h, col_off:col_off+w]
        score_tile = solar_score[row_off:row_off+h, col_off:col_off+w]

        # Flatten tile
        ndvi_flat = ndvi_tile.flatten()
        cloud_flat = cloud_tile.flatten()
        score_flat = score_tile.flatten()

        # Sample random pixeli
        n = min(sample_per_tile, len(ndvi_flat))
        indices = np.random.choice(len(ndvi_flat), size=n, replace=False)

        X_list.append(np.stack([ndvi_flat[indices], cloud_flat[indices]], axis=1))
        y_list.append(score_flat[indices])

# Concatenăm toate datele
X = np.vstack(X_list).astype(np.float32)
y = np.hstack(y_list).astype(np.float32)

print("Dimensiune dataset ML:", X.shape)

# -----------------------------
# Împărțim în train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Model Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

# Evaluare
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE pe setul de test:", mse)

# -----------------------------
# Predicție pe întreaga zonă
X_all = np.stack([ndvi.flatten(), cloud_fraction.flatten()], axis=1).astype(np.float32)
pred_full = rf.predict(X_all).reshape(ndvi.shape)

# Salvare GeoTIFF
safe_folder = sorted(glob.glob("downloaded data/*.SAFE"))[0]
red_path = glob.glob(os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA", "R10m", "*_B04_10m.jp2"))[0]

with rasterio.open(red_path) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, driver='GTiff', compress='lzw', height=pred_full.shape[0], width=pred_full.shape[1])

output_file = "solar_score_ml.tif"
with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(pred_full.astype(np.float32), 1)

print(f"✅ Harta estimată de ML (tile sampling) salvată ca {output_file}")
