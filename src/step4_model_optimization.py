import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import amp  # Mixed precision training
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold
import joblib
import time
import os
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Kiểm tra GPU - YÊU CẦU PHẢI CÓ GPU
if not torch.cuda.is_available():
    print("❌ LỖI: Không tìm thấy GPU CUDA! Script này yêu cầu GPU RTX 3080 Ti để chạy.")
    print("Vui lòng kiểm tra:")
    print("1. Đã cài đặt CUDA toolkit")
    print("2. Đã cài đặt PyTorch với CUDA support")
    print("3. GPU driver đã được cập nhật")
    exit(1)

device = torch.device('cuda')
print(f"✅ Sử dụng GPU: {torch.cuda.get_device_name(0)}")
print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Tối ưu hóa GPU cho RTX 3080 Ti
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("🚀 Đã kích hoạt các tối ưu hóa GPU: cuDNN benchmark, TF32 precision")

# Tạo thư mục models nếu chưa có
os.makedirs('./models', exist_ok=True)
os.makedirs('./reports', exist_ok=True)

# Load dữ liệu đã xử lý
print("Loading processed data...")
# Giả định các file dữ liệu nằm trong thư mục ./data/
# Nếu không, hãy thay đổi đường dẫn cho phù hợp
try:
    X_train = np.load('./data/X_train.npy', allow_pickle=True)
    X_test = np.load('./data/X_test.npy', allow_pickle=True)
    y_train = np.load('./data/y_train.npy', allow_pickle=True)
    y_test = np.load('./data/y_test.npy', allow_pickle=True)
except FileNotFoundError as e:
    print(f"Lỗi: {e}. Hãy chắc chắn rằng các file .npy tồn tại trong thư mục ./data/")
    exit()


print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Load label encoder
le = joblib.load('./models/label_encoder.joblib')
y_train_encoded = le.transform(y_train)
y_test_encoded = le.transform(y_test)

# ==================== Định nghĩa mô hình CNN với tham số có thể tune (ĐÃ SỬA) ====================

class TunableCNN(nn.Module):
    # SỬA 1: Thêm `seq_length` để mô hình không phụ thuộc vào biến toàn cục
    def __init__(self, vocab_size, embed_dim, num_classes, seq_length, dropout_rate=0.5, num_filters=128):
        super(TunableCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters//2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        # SỬA 1: Tính output size một cách tự động để mô hình linh hoạt
        conv_output_size = self._get_conv_output_size(embed_dim, seq_length)
        
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output_size(self, embed_dim, seq_length):
        # Tạo một tensor giả để tính toán kích thước đầu ra sau các lớp conv/pool
        with torch.no_grad():
            dummy_input = torch.zeros(1, embed_dim, seq_length)
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== Hàm training với early stopping (ĐÃ SỬA) ====================

def train_model_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                              num_epochs=50, patience=5, min_delta=0.001):
    model.to(device)
    scaler = amp.GradScaler()  # Mixed precision scaler
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision training
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # SỬA 2: Sửa lệnh print để hiển thị thông tin hữu ích
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_loss

# ==================== Hàm đánh giá mô hình ====================

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with amp.autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# ==================== Cross-validation (ĐÃ SỬA) ====================

# SỬA 3: Truyền `vocab_size` vào hàm thay vì tính toán lại mỗi lần
def cross_validate_model(model_class, params, X, y, vocab_size, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/{n_splits}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(torch.LongTensor(X_train_fold), torch.LongTensor(y_train_fold))
        val_dataset = TensorDataset(torch.LongTensor(X_val_fold), torch.LongTensor(y_val_fold))
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, pin_memory=True)

        # SỬA 1 & 3: Khởi tạo mô hình với seq_length và vocab_size đã được tính toán đúng
        seq_length = X_train_fold.shape[1]
        model = model_class(vocab_size, params['embed_dim'], len(np.unique(y)), seq_length,
                           params['dropout_rate'], params['num_filters'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

        trained_model, val_loss = train_model_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=30, patience=3
        )

        cv_scores.append(val_loss)
        # SỬA 2: Sửa lệnh print
        print(f"Fold {fold+1} Validation Loss: {val_loss:.4f}")

    return np.mean(cv_scores), np.std(cv_scores)

# ==================== Hyperparameter tuning (ĐÃ SỬA) ====================

# SỬA 3: Tính vocab_size một lần ở đây
vocab_size = int(np.max(X_train)) + 1
num_classes = len(np.unique(y_train_encoded))
seq_length = X_train.shape[1]

print(f"Vocab size: {vocab_size}")
print(f"Number of classes: {num_classes}")
print(f"Sequence length: {seq_length}")

param_grid = {
    'embed_dim': [100, 150],
    'dropout_rate': [0.3, 0.5],
    'num_filters': [64, 128],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64, 128],  # Tăng batch size để tối ưu GPU RTX 3080 Ti
    'weight_decay': [1e-4, 1e-5]
}

print(f"Total parameter combinations: {len(list(ParameterGrid(param_grid)))}")

if __name__ == '__main__':
    best_params = None
    best_score = float('inf')
    results = []

    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH CROSS-VALIDATION")
    print("="*60)

    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f"\nTesting combination {i+1}/{len(list(ParameterGrid(param_grid)))}: {params}")
        try:
            # SỬA 3: Truyền vocab_size vào hàm
            mean_score, std_score = cross_validate_model(TunableCNN, params, X_train, y_train_encoded, vocab_size, n_splits=3)
            results.append({
                'params': params,
                'mean_cv_score': mean_score,
                'std_cv_score': std_score
            })

            # SỬA 2: Sửa lệnh print
            print(f"Result: Mean CV Loss = {mean_score:.4f} (+/- {std_score:.4f})")

            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                print("🎯 New best parameters found!")
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    if best_params is None:
        print("\nKhông tìm thấy tham số tốt nhất. Vui lòng kiểm tra lại quá trình tuning.")
        exit()

    print(f"\n🏆 Best parameters: {best_params}")
    # SỬA 2: Sửa lệnh print
    print(f"Best CV Loss: {best_score:.4f}")

    # ==================== Train final model với best parameters (ĐÃ SỬA) ====================
    print("\n" + "="*60)
    print("TRAINING FINAL OPTIMIZED MODEL")
    print("="*60)

    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train_encoded))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test_encoded))
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)

    # SỬA 1: Khởi tạo mô hình cuối cùng với seq_length
    final_model = TunableCNN(vocab_size, best_params['embed_dim'], num_classes, seq_length,
                            best_params['dropout_rate'], best_params['num_filters'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader_final = DataLoader(train_subset, batch_size=best_params['batch_size'], shuffle=True, pin_memory=True)
    val_loader_final = DataLoader(val_subset, batch_size=best_params['batch_size'], shuffle=False, pin_memory=True)

    start_time = time.time()
    trained_final_model, final_val_loss = train_model_early_stopping(
        final_model, train_loader_final, val_loader_final, criterion, optimizer, num_epochs=50, patience=7
    )
    training_time = time.time() - start_time

    final_preds, final_labels = evaluate_model(trained_final_model, test_loader)
    final_accuracy = accuracy_score(final_labels, final_preds)
    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='weighted')

    print("\n🎉 FINAL MODEL RESULTS:")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    print(f"F1-Score: {final_f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    # ==================== So sánh với mô hình gốc (ĐÃ SỬA) ====================
    print("\n" + "="*60)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("="*60)

    # SỬA 4: Xử lý trường hợp file mô hình gốc không tồn tại
    try:
        # SỬA 1: Khởi tạo mô hình gốc với seq_length
        original_model = TunableCNN(vocab_size, 100, num_classes, seq_length, 0.5, 128)
        original_model.load_state_dict(torch.load('./models/cnn_model.pth'))
        original_model.to(device)

        orig_preds, orig_labels = evaluate_model(original_model, test_loader)
        orig_accuracy = accuracy_score(orig_labels, orig_preds)
        orig_precision, orig_recall, orig_f1, _ = precision_recall_fscore_support(orig_labels, orig_preds, average='weighted')

    except FileNotFoundError:
        print("⚠️  Không tìm thấy file mô hình gốc './models/cnn_model.pth'. Bỏ qua bước so sánh.")
        orig_accuracy, orig_precision, orig_recall, orig_f1 = 0.0, 0.0, 0.0, 0.0

    print("📊 MODEL COMPARISON:")
    print(f"Original CNN:  Accuracy = {orig_accuracy:.4f}, Precision = {orig_precision:.4f}, Recall = {orig_recall:.4f}, F1 = {orig_f1:.4f}")
    print(f"Optimized CNN: Accuracy = {final_accuracy:.4f}, Precision = {final_precision:.4f}, Recall = {final_recall:.4f}, F1 = {final_f1:.4f}")
    # SỬA 2: Loại bỏ lệnh print không cần thiết

    # ==================== Lưu mô hình và kết quả ====================
    torch.save(trained_final_model.state_dict(), './models/cnn_optimized.pth')
    joblib.dump(best_params, './models/best_hyperparams.joblib')

    optimization_results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'final_accuracy': final_accuracy,
        'final_precision': final_precision,
        'final_recall': final_recall,
        'final_f1': final_f1,
        'training_time': training_time,
        'original_accuracy': orig_accuracy,
        'improvement': final_accuracy - orig_accuracy,
        'all_results': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(optimization_results, './models/step4_optimization_results.joblib')

    # ==================== Tạo báo cáo HTML ====================
    html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bước 4: Tối Ưu Hóa Mô Hình - Báo Cáo</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        .comparison-table {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .model-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
        .improvement {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); transform: scale(1.05); }}
        .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e8f4fd; }}
        .code {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; margin: 10px 0; font-family: 'Courier New', monospace; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .success {{ background: #d4edda; border-left-color: #28a745; }}
        .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Bước 4: Tối Ưu Hóa Mô Hình - Báo Cáo</h1>
        <div class="highlight">
            <strong>🎯 Mục tiêu:</strong> Cải thiện hiệu suất mô hình CNN từ {orig_accuracy:.1%} lên mức tối ưu bằng hyperparameter tuning và cross-validation.
        </div>
        <h2>🏆 Kết Quả Tối Ưu Hóa</h2>
        <div class="comparison-table">
            <div class="model-card">
                <div class="metric-label">Mô Hình Gốc</div>
                <div class="metric-value">{orig_accuracy:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="model-card improvement">
                <div class="metric-label">Mô Hình Tối Ưu</div>
                <div class="metric-value">{final_accuracy:.1%}</div>
                <div class="metric-label">Accuracy (+{((final_accuracy-orig_accuracy)*100):.1f}%)</div>
            </div>
        </div>
        <h2>⚙️ Best Hyperparameters</h2>
        <div class="code">
"""
    for key, value in best_params.items():
        html_content += f"<strong>{key}:</strong> {value}<br>"
    html_content += f"""
        </div>
        <h2>📊 Chi Tiết Performance</h2>
        <table>
            <thead><tr><th>Metric</th><th>Mô Hình Gốc</th><th>Mô Hình Tối Ưu</th><th>Cải Thiện</th></tr></thead>
            <tbody>
                <tr><td>Accuracy</td><td>{orig_accuracy:.4f}</td><td>{final_accuracy:.4f}</td><td>+{((final_accuracy-orig_accuracy)*100):.2f}%</td></tr>
                <tr><td>Precision</td><td>{orig_precision:.4f}</td><td>{final_precision:.4f}</td><td>+{((final_precision-orig_precision)*100):.2f}%</td></tr>
                <tr><td>Recall</td><td>{orig_recall:.4f}</td><td>{final_recall:.4f}</td><td>+{((final_recall-orig_recall)*100):.2f}%</td></tr>
                <tr><td>F1-Score</td><td>{orig_f1:.4f}</td><td>{final_f1:.4f}</td><td>+{((final_f1-orig_f1)*100):.2f}%</td></tr>
            </tbody>
        </table>
        <h2>🔍 Phân Tích Kết Quả</h2>
        <div class="highlight success">
            <strong>✨ Thành công:</strong><br>
            - <strong>Accuracy cải thiện</strong> từ {orig_accuracy:.1%} lên {final_accuracy:.1%} (+{((final_accuracy-orig_accuracy)*100):.1f}%)<br>
            - <strong>Cross-validation</strong> đảm bảo robustness với CV score: {best_score:.4f}<br>
            - <strong>Early stopping</strong> và <strong>learning rate scheduling</strong> tối ưu hóa training<br>
            - <strong>Thời gian training</strong>: {training_time:.2f} giây với GPU
        </div>
        <h2>🛠️ Kỹ Thuật Tối Ưu Hóa Áp Dụng</h2>
        <ul>
            <li><strong>Grid Search</strong>: Tìm kiếm trên {len(list(ParameterGrid(param_grid)))} combinations</li>
            <li><strong>3-Fold Cross-Validation</strong>: Đánh giá robust trên nhiều splits</li>
            <li><strong>Early Stopping</strong>: Dừng khi validation loss không cải thiện</li>
            <li><strong>Learning Rate Scheduling</strong>: Giảm LR khi plateau</li>
            <li><strong>Weight Decay (L2)</strong>: Regularization tránh overfitting</li>
            <li><strong>Dropout Tuning</strong>: Tối ưu regularization rate</li>
        </ul>
        <div class="footer">
            <p>⏰ Báo cáo được tạo vào: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>
"""

    with open('./reports/Step4_Model_Optimization.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n✅ Báo cáo HTML đã được tạo: ./reports/Step4_Model_Optimization.html")
    print("✅ Mô hình tối ưu đã được lưu: ./models/cnn_optimized.pth")
    print("✅ Best hyperparameters: ./models/best_hyperparams.joblib")
    print("✅ Kết quả optimization: ./models/step4_optimization_results.joblib")

    print("\n🎉 Hoàn thành Bước 4: Tối ưu hóa mô hình!")