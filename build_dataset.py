import pandas as pd
import numpy as np
import os

# تحميل البيانات
df = pd.read_csv("data/points_indicators.csv")

# تأكيد الترتيب
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['point_id', 'date'])

# تحديد التواريخ والمؤشرات
dates = df['date'].sort_values().unique()
indices = sorted([col for col in df.columns if col not in ['point_id', 'date', 'label']])
points = df['point_id'].unique()

print(f"🧩 Total Points: {len(points)}")
print(f"📅 Dates: {len(dates)}")
print(f"🛰️ Indicators: {indices}")

# بناء X و y
X = np.zeros((len(points), len(dates), len(indices)), dtype=np.float32)
y = np.zeros(len(points), dtype=np.int32)

# بناء القاموس المؤقت
for i, pid in enumerate(points):
    sub = df[df['point_id'] == pid]
    sub = sub.sort_values(by='date')
    
    if len(sub) != len(dates):
        print(f"⚠️ Point {pid} has {len(sub)} records instead of {len(dates)}")
        continue
    
    X[i] = sub[indices].values
    y[i] = sub['label'].iloc[0]

# حفظ
os.makedirs("output", exist_ok=True)
np.save("output/X_timeseries.npy", X)
np.save("output/y_labels.npy", y)
print("✅ Saved: output/X_timeseries.npy, output/y_labels.npy")
