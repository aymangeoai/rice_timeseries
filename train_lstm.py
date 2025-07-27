import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# تحميل البيانات
X = np.load('../output/X_timeseries.npy')  # الشكل: [عينات, تواريخ, مؤشرات]
y = np.load('../output/y_labels.npy')      # الشكل: [عينات]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء نموذج LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# حفظ أفضل نموذج
if not os.path.exists('../models'):
    os.makedirs('../models')

checkpoint = ModelCheckpoint('../models/lstm.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[checkpoint])

# التقييم
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
