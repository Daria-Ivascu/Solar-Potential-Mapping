import glob
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

safe_folders = sorted(glob.glob("downloaded data/*.SAFE"))

tile_size = 1000

cloud_sum = None
count = 0

os.makedirs("ndvi_tiles", exist_ok=True)

for idx, safe_folder in enumerate(tqdm(safe_folders, desc="Procesăm SAFE files")):
    # RED and NIR bands
    band_paths = sorted(glob.glob(os.path.join(
        safe_folder, "GRANULE", "*", "IMG_DATA", "R10m", "*_10m.jp2"
    )))
    red_path = [p for p in band_paths if "B04" in p][0]
    nir_path = [p for p in band_paths if "B08" in p][0]

    scl_path = glob.glob(os.path.join(
        safe_folder, "GRANULE", "*", "IMG_DATA", "R20m", "*_SCL_20m.jp2"
    ))[0]

    with rasterio.open(red_path) as red_src, \
         rasterio.open(nir_path) as nir_src, \
         rasterio.open(scl_path) as scl_src:

        height, width = red_src.height, red_src.width

        if cloud_sum is None:
            cloud_sum = np.zeros((height, width), dtype=np.float32)

        tiles_y = (height + tile_size - 1) // tile_size
        tiles_x = (width + tile_size - 1) // tile_size

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                row_off = ty * tile_size
                col_off = tx * tile_size
                h = min(tile_size, height - row_off)
                w = min(tile_size, width - col_off)

                if h <= 0 or w <= 0:
                    continue

                win = rasterio.windows.Window(col_off, row_off, w, h)

                red = red_src.read(1, window=win).astype(np.float32) / 10000.0
                nir = nir_src.read(1, window=win).astype(np.float32) / 10000.0
                ndvi = (nir - red) / (nir + red + 1e-6)

                # Resample SCL from 20m to 10m resolution to align with NDVI
                row_off_scl = row_off // 2
                col_off_scl = col_off // 2
                h_scl = (h + 1) // 2
                w_scl = (w + 1) // 2

                win_scl = rasterio.windows.Window(col_off_scl, row_off_scl, w_scl, h_scl)

                scl = scl_src.read(
                    1,
                    window=win_scl,
                    out_shape=(h, w),   # resample to 10m
                    resampling=Resampling.nearest
                )

                cloud_mask = np.isin(scl, [3, 8, 9, 10]).astype(np.float32)

                np.save(f"ndvi_tiles/ndvi_{idx}_{ty}_{tx}.npy", ndvi)

                cloud_sum[row_off:row_off + h, col_off:col_off + w] += cloud_mask

    count += 1

# Cloud fraction final
cloud_fraction = cloud_sum / count
np.save("cloud_fraction.npy", cloud_fraction)

# NDVI median per pixel
ndvi_median = np.full_like(cloud_fraction, np.nan, dtype=np.float32)

tiles_y = (cloud_fraction.shape[0] + tile_size - 1) // tile_size
tiles_x = (cloud_fraction.shape[1] + tile_size - 1) // tile_size

for ty in tqdm(range(tiles_y), desc="Calculăm median NDVI"):
    for tx in range(tiles_x):
        ndvi_list = []
        for idx in range(len(safe_folders)):
            path = f"ndvi_tiles/ndvi_{idx}_{ty}_{tx}.npy"
            if os.path.exists(path):
                ndvi_list.append(np.load(path))

        if ndvi_list:
            tile_stack = np.stack(ndvi_list, axis=0)
            median_tile = np.median(tile_stack, axis=0)

            row_off = ty * tile_size
            col_off = tx * tile_size
            h, w = median_tile.shape

            ndvi_median[row_off:row_off + h, col_off:col_off + w] = median_tile

np.save("ndvi_median.npy", ndvi_median)

print("NDVI median saved as ndvi_median.npy")
print("Cloud fraction saved as cloud_fraction.npy")
