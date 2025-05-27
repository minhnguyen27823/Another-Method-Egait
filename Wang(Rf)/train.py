import os
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Cấu hình
csv_folder = "data_process/"
labels_path = "labels_my.h5"
WINDOW_SIZE = 30  # số frame mỗi sample

# Kết quả
samples = []
labels = []

# Mở file nhãn
with h5py.File(labels_path, 'r') as labels_file:
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(csv_folder, filename)
            key = os.path.splitext(filename)[0]

            if key not in labels_file:
                print(f"[!] Không có label cho {key}, bỏ qua.")
                continue

            try:
                # Đọc dữ liệu CSV
                df = pd.read_csv(filepath)
                data_array = df.values  # chuyển về numpy array
                num_rows = data_array.shape[0]

                # Lấy label (giả sử là scalar)
                label = labels_file[key][()]
                if isinstance(label, bytes):
                    label = label.decode()

                # Trích các đoạn 30 dòng
                for start in range(0, num_rows - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window = data_array[start:start + WINDOW_SIZE]
                    samples.append(window)
                    labels.append(label)

            except Exception as e:
                print(f"[!] Lỗi khi xử lý {filename}: {e}")



X = np.array(samples)       # shape: (num_samples, 30, num_features)
y = np.array(labels)        # shape: (num_samples,)

print("✅ Tổng số samples:", len(X))
print("Shape X:", X.shape)
print("Shape y:", y.shape)
num_samples, time_steps, num_features = X.shape
X_flat = X.reshape(num_samples, time_steps * num_features)

X_train, X_test, y_train, y_test = train_test_split( X_flat, y, test_size=0.2, random_state=42, stratify=y)

# Khởi tạo mô hình
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện
rf.fit(X_train, y_train)

# Dự đoán
y_pred = rf.predict(X_test)

# Đánh giá
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
