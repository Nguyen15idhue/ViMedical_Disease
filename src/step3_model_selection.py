import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier  # XGBoost alternative in sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import time
import joblib

print("=== Bước 3: Chọn và So Sánh 3 Thuật Toán Tối Ưu ===")
print("Đang tải dữ liệu đã xử lý từ Bước 2...")

# Load processed data from Step 2
X_train = np.load('../data/X_train.npy', allow_pickle=True)
y_train_str = np.load('../data/y_train.npy', allow_pickle=True)
X_test = np.load('../data/X_test.npy', allow_pickle=True)
y_test_str = np.load('../data/y_test.npy', allow_pickle=True)

print("Dữ liệu đã tải thành công.")
print(f"Kích thước tập train: {X_train.shape}, tập test: {X_test.shape}")

# Encode labels to int
print("Đang mã hóa nhãn thành số nguyên...")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_str)
y_test = label_encoder.transform(y_test_str)
print("Mã hóa nhãn hoàn tất.")

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

print("Đang chuyển đổi dữ liệu thành PyTorch tensors...")
# Convert to PyTorch tensors for GPU if available
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
print("Chuyển đổi tensors hoàn tất.")

# Note: For simplicity, we'll use sklearn models as PyTorch doesn't have built-in RF/SVM/XGBoost
# But we'll move data to GPU tensors (though sklearn doesn't use them directly, it's for consistency)
# XGBoost can use GPU, but here we'll use sklearn GradientBoosting as proxy for XGBoost

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', C=1.0, random_state=42),
    'XGBoost (Gradient Boosting)': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Đang huấn luyện mô hình: {name} ---")
    start_time = time.time()

    print(f"Khởi tạo và fit mô hình {name}...")
    # Fit model (sklearn handles GPU for XGBoost if configured, but here using CPU for simplicity)
    model.fit(X_train, y_train)  # Use numpy arrays for sklearn

    train_time = time.time() - start_time
    print(".2f")

    print(f"Đang dự đoán trên tập test với {name}...")
    # Predict
    y_pred = model.predict(X_test)

    print(f"Tính toán metrics cho {name}...")
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Train Time': train_time,
        'Model': model
    }

    print(f"Kết quả {name}:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

print("\n--- Tìm mô hình tốt nhất ---")
# Save best model (highest accuracy)
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = results[best_model_name]['Model']
joblib.dump(best_model, '../data/best_model.pkl')
print(f"Mô hình tốt nhất: {best_model_name} với độ chính xác {results[best_model_name]['Accuracy']:.4f}")
print("Mô hình tốt nhất đã lưu vào ../data/best_model.pkl")

# Print comparison table
print("\n=== Bảng So Sánh Kết Quả ===")
print("| Thuật Toán | Độ Chính Xác | Precision | Recall | F1-Score | Thời Gian Train (s) |")
print("|------------|---------------|-----------|--------|----------|---------------------|")
for name, res in results.items():
    print(".4f")

print("\n--- Báo Cáo Chi Tiết Cho Mô Hình Tốt Nhất ---")
# Detailed classification report for best model
print(f"Báo cáo chi tiết cho {best_model_name}:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=[f'Lớp {i}' for i in range(len(np.unique(y_test)))]))

print("\n=== Bước 3 Hoàn Thành: Các Mô Hình Đã Được Huấn Luyện và So Sánh ===")
print("Bạn có thể xem kết quả trong bảng trên và mô hình đã lưu.")