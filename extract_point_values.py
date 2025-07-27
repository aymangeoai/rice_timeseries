import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import Point
from tqdm import tqdm

# 🔧 المسارات
points_path = "data/rice_points.shp"
raster_dir = "data/processed"

# تحميل النقاط
gdf = gpd.read_file(points_path)
gdf = gdf.to_crs(epsg=32636)  # تأكد أن CRS يتطابق مع الصور

# إعداد الإطار الزمني
raster_files = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
dates = sorted(list(set([os.path.basename(f).split("_")[0] for f in raster_files])))
indices = sorted(list(set([os.path.basename(f).split("_")[1].split(".")[0] for f in raster_files])))

# التجهيز
records = []

for date in tqdm(dates, desc="🧪 Extracting per date"):
    row = {'date': date}
    rasters = {}

    # تحميل كل المؤشرات لهذا التاريخ
    for idx in indices:
        fpath = os.path.join(raster_dir, f"{date}_{idx}.tif")
        if os.path.exists(fpath):
            with rasterio.open(fpath) as src:
                rasters[idx] = (src.read(1), src.transform)
        else:
            print(f"⚠️ Missing {fpath}")

    # لكل نقطة
    for i, pt in gdf.iterrows():
        px, py = pt.geometry.x, pt.geometry.y
        point_data = {'point_id': pt['id'], 'date': date, 'label': pt['label']}  # label: 0 or 1

        for idx in indices:
            if idx not in rasters:
                point_data[idx] = None
                continue
            arr, transform = rasters[idx]
            col, row = ~transform * (px, py)
            col, row = int(col), int(row)
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                point_data[idx] = float(arr[row, col])
            else:
                point_data[idx] = None

        records.append(point_data)

# حفظ إلى CSV
df = pd.DataFrame(records)
df.to_csv("data/points_indicators.csv", index=False)
print("✅ Saved: data/points_indicators.csv")
