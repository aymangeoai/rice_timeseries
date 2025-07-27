import os
import rasterio
import numpy as np
from glob import glob
from rasterio.enums import Resampling
from rasterio import Affine
from tqdm import tqdm

# 🛠️ تعيين المسارات
input_dir = "data/sentinel_raw"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# 🧮 دوال حساب المؤشرات الطيفية
def ndvi(b8, b4): return (b8 - b4) / (b8 + b4 + 1e-10)
def lswi(b8, b11): return (b8 - b11) / (b8 + b11 + 1e-10)
def ndwi(b3, b8): return (b3 - b8) / (b3 + b8 + 1e-10)
def evi(b8, b4, b2): return 2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1)
def savi(b8, b4): return 1.5 * (b8 - b4) / (b8 + b4 + 0.5)
def ndre(b8, b5): return (b8 - b5) / (b8 + b5 + 1e-10)
def gci(b8, b3): return (b8 / (b3 + 1e-10)) - 1
def msavi(b8, b4): return (2 * b8 + 1 - np.sqrt((2*b8 + 1)**2 - 8*(b8 - b4))) / 2
def ndbi(b11, b8): return (b11 - b8) / (b11 + b8 + 1e-10)

# 🧾 المؤشرات المطلوبة مع القنوات المطلوبة
indices = {
    "NDVI": (ndvi, ["B8", "B4"]),
    "LSWI": (lswi, ["B8", "B11"]),
    "NDWI": (ndwi, ["B3", "B8"]),
    "EVI": (evi, ["B8", "B4", "B2"]),
    "SAVI": (savi, ["B8", "B4"]),
    "NDRE": (ndre, ["B8", "B5"]),
    "GCI": (gci, ["B8", "B3"]),
    "MSAVI": (msavi, ["B8", "B4"]),
    "NDBI": (ndbi, ["B11", "B8"]),
}

# 🔁 معالجة كل مجلد صورة
for scene_folder in tqdm(sorted(os.listdir(input_dir)), desc="Processing Scenes"):
    scene_path = os.path.join(input_dir, scene_folder)
    if not os.path.isdir(scene_path):
        continue

    bands = {}
    profile = None

    # تحميل القنوات
    for band_name in ["B2", "B3", "B4", "B5", "B8", "B11"]:
        band_path = os.path.join(scene_path, f"{band_name}.tif")
        if not os.path.exists(band_path):
            print(f"❌ Missing {band_name} in {scene_folder}")
            continue

        with rasterio.open(band_path) as src:
            bands[band_name] = src.read(1).astype('float32')
            if profile is None:
                profile = src.profile

    if len(bands) < 6:
        print(f"⚠️ Skipping {scene_folder} due to missing bands")
        continue

    # حساب كل مؤشر وحفظه
    for index_name, (func, required_bands) in indices.items():
        try:
            arrays = [bands[b] for b in required_bands]
            result = func(*arrays)
            result = np.clip(result, -1, 1)

            out_path = os.path.join(output_dir, f"{scene_folder}_{index_name}.tif")
            profile.update(dtype='float32', count=1)

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(result, 1)
        except Exception as e:
            print(f"⚠️ Failed to process {index_name} for {scene_folder}: {e}")
