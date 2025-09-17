import glob
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import rasterio

tile_size = 1000
sample_per_tile = 5000
dtype = np.float32

# Load NDVI
ndvi = np.load("ndvi_median.npy")
height, width = ndvi.shape

# Input raster layers
rasters = {
    "cloud_fraction": "cloud_fraction.npy",
    "slope": "slope.npy",
    "elevation": "elevation.npy",
    "solar_score": "solar_score.npy"
}

# Resize rasters to match NDVI grid
def resize_blockwise(src_array, target_shape, tile_size=1000):
    height, width = target_shape
    dst_array = np.empty((height, width), dtype=np.float32)
    
    scale_y = target_shape[0] / src_array.shape[0]
    scale_x = target_shape[1] / src_array.shape[1]
    
    for y_off in range(0, height, tile_size):
        h = min(tile_size, height - y_off)
        for x_off in range(0, width, tile_size):
            w = min(tile_size, width - x_off)
            
            src_y0 = int(y_off / scale_y)
            src_y1 = int((y_off + h) / scale_y)
            src_x0 = int(x_off / scale_x)
            src_x1 = int((x_off + w) / scale_x)
            
            src_block = src_array[src_y0:src_y1, src_x0:src_x1]
            
            dst_block = cv2.resize(src_block, (w, h), interpolation=cv2.INTER_LINEAR)
            dst_array[y_off:y_off+h, x_off:x_off+w] = dst_block
            
    return dst_array.astype(np.float32)

# Resize all predictors to NDVI grid
resized_layers = {}
for name, npy_file in rasters.items():
    src_array = np.load(npy_file).astype(np.float32)
    resized_layers[name] = resize_blockwise(src_array, ndvi.shape, tile_size=tile_size)
    del src_array

# Prepare training data
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

        # Extract tiles for all variables
        ndvi_tile = ndvi[row_off:row_off+h, col_off:col_off+w]
        cloud_tile = resized_layers['cloud_fraction'][row_off:row_off+h, col_off:col_off+w]
        slope_tile = resized_layers['slope'][row_off:row_off+h, col_off:col_off+w]
        elev_tile = resized_layers['elevation'][row_off:row_off+h, col_off:col_off+w]
        score_tile = resized_layers['solar_score'][row_off:row_off+h, col_off:col_off+w]

        # Flatten arrays
        ndvi_flat = ndvi_tile.flatten()
        cloud_flat = cloud_tile.flatten()
        slope_flat = slope_tile.flatten()
        elev_flat = elev_tile.flatten()
        score_flat = score_tile.flatten()

        # Randomly sample pixels from each tile
        n = min(sample_per_tile, len(ndvi_flat))
        indices = np.random.choice(len(ndvi_flat), size=n, replace=False)

        # Stack predictors
        X_list.append(
            np.stack([
                ndvi_flat[indices],
                cloud_flat[indices],
                slope_flat[indices],
                elev_flat[indices]
            ], axis=1)
        )

        # Training target
        y_list.append(score_flat[indices])

# Final training dataset
X = np.vstack(X_list).astype(np.float32)
y = np.hstack(y_list).astype(np.float32)
del X_list, y_list
print("Dimensiune dataset ML:", X.shape)

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X, y

# Random Forest model
rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE pe setul de test:", mse)

# Predict full map block by block
pred_full = np.empty(ndvi.shape, dtype=np.float32)

for y_off in tqdm(range(0, height, tile_size), desc="Predicting full map"):
    h = min(tile_size, height - y_off)
    for x_off in range(0, width, tile_size):
        w = min(tile_size, width - x_off)

        # Stack features for current block
        ndvi_tile = ndvi[y_off:y_off+h, x_off:x_off+w]
        cloud_tile = resized_layers['cloud_fraction'][y_off:y_off+h, x_off:x_off+w]
        slope_tile = resized_layers['slope'][y_off:y_off+h, x_off:x_off+w]
        elev_tile = resized_layers['elevation'][y_off:y_off+h, x_off:x_off+w]

        X_block = np.stack([
            ndvi_tile.flatten(),
            cloud_tile.flatten(),
            slope_tile.flatten(),
            elev_tile.flatten()
        ], axis=1).astype(np.float32)

        # Predict solar score of the block
        pred_block = rf.predict(X_block).reshape((h, w))
        pred_full[y_off:y_off+h, x_off:x_off+w] = pred_block

# Get georeference info from Sentinel-2 JP2 band
safe_folder = sorted(glob.glob("downloaded data/*.SAFE"))[0]
red_path = glob.glob(os.path.join(safe_folder, "GRANULE", "*", "IMG_DATA", "R10m", "*_B04_10m.jp2"))[0]

with rasterio.open(red_path) as src:
    profile = src.profile
    profile.update(dtype=rasterio.float32, count=1, driver='GTiff',
                   compress='lzw', height=pred_full.shape[0], width=pred_full.shape[1])

# Save prediction
output_file = "solar_score_ml.tif"
with rasterio.open(output_file, "w", **profile) as dst:
    dst.write(pred_full, 1)

print(f"Solar potential map estimated with ML saved as {output_file}")
