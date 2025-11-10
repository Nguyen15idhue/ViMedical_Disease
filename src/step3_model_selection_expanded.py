import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Tạo thư mục models nếu chưa có
os.makedirs('../models', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

# Load dữ liệu đã xử lý
print("Loading processed data...")
X_train = np.load('../data/X_train.npy', allow_pickle=True)
X_test = np.load('../data/X_test.npy', allow_pickle=True)
y_train = np.load('../data/y_train.npy', allow_pickle=True)
y_test = np.load('../data/y_test.npy', allow_pickle=True)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Chuyển đổi labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Lưu label encoder
joblib.dump(le, '../models/label_encoder.joblib')

# ==================== Định nghĩa các mô hình ====================

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (X_train.shape[1] // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

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

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Lấy output của timestep cuối
        x = self.dropout(x)
        x = self.fc(x)
        return x

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Lấy output của timestep cuối
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== Hàm huấn luyện ====================

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

# ==================== Thiết lập tham số ====================

vocab_size = int(np.max(X_train)) + 1
embed_dim = 100
num_classes = len(np.unique(y_train_encoded))
batch_size = 32
num_epochs = 10

print(f"Vocab size: {vocab_size}")
print(f"Embedding dim: {embed_dim}")
print(f"Number of classes: {num_classes}")

# Tạo DataLoader
train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train_encoded))
test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test_encoded))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== Huấn luyện các mô hình ====================

models_results = {}

# 1. CNN
print("\n" + "="*50)
print("Training CNN...")
cnn_model = CNNModel(vocab_size, embed_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

start_time = time.time()
trained_cnn = train_model(cnn_model, train_loader, criterion, optimizer, num_epochs)
cnn_time = time.time() - start_time

cnn_preds, cnn_labels = evaluate_model(trained_cnn, test_loader)
cnn_accuracy = accuracy_score(cnn_labels, cnn_preds)
cnn_precision, cnn_recall, cnn_f1, _ = precision_recall_fscore_support(cnn_labels, cnn_preds, average='weighted')

models_results['CNN'] = {
    'accuracy': cnn_accuracy,
    'precision': cnn_precision,
    'recall': cnn_recall,
    'f1': cnn_f1,
    'time': cnn_time,
    'predictions': cnn_preds,
    'true_labels': cnn_labels
}

# Lưu mô hình CNN
torch.save(trained_cnn.state_dict(), '../models/cnn_model.pth')
print(".4f")

# 2. LSTM
print("\n" + "="*50)
print("Training LSTM...")
lstm_model = LSTMModel(vocab_size, embed_dim, num_classes)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

start_time = time.time()
trained_lstm = train_model(lstm_model, train_loader, criterion, optimizer, num_epochs)
lstm_time = time.time() - start_time

lstm_preds, lstm_labels = evaluate_model(trained_lstm, test_loader)
lstm_accuracy = accuracy_score(lstm_labels, lstm_preds)
lstm_precision, lstm_recall, lstm_f1, _ = precision_recall_fscore_support(lstm_labels, lstm_preds, average='weighted')

models_results['LSTM'] = {
    'accuracy': lstm_accuracy,
    'precision': lstm_precision,
    'recall': lstm_recall,
    'f1': lstm_f1,
    'time': lstm_time,
    'predictions': lstm_preds,
    'true_labels': lstm_labels
}

# Lưu mô hình LSTM
torch.save(trained_lstm.state_dict(), '../models/lstm_model.pth')
print(".4f")

# 3. GRU
print("\n" + "="*50)
print("Training GRU...")
gru_model = GRUModel(vocab_size, embed_dim, num_classes)
optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

start_time = time.time()
trained_gru = train_model(gru_model, train_loader, criterion, optimizer, num_epochs)
gru_time = time.time() - start_time

gru_preds, gru_labels = evaluate_model(trained_gru, test_loader)
gru_accuracy = accuracy_score(gru_labels, gru_preds)
gru_precision, gru_recall, gru_f1, _ = precision_recall_fscore_support(gru_labels, gru_preds, average='weighted')

models_results['GRU'] = {
    'accuracy': gru_accuracy,
    'precision': gru_precision,
    'recall': gru_recall,
    'f1': gru_f1,
    'time': gru_time,
    'predictions': gru_preds,
    'true_labels': gru_labels
}

# Lưu mô hình GRU
torch.save(trained_gru.state_dict(), '../models/gru_model.pth')
print(".4f")

# 4. XGBoost với embeddings
print("\n" + "="*50)
print("Training XGBoost with embeddings...")

# Tạo embeddings đơn giản từ dữ liệu
def create_embeddings(X_data, embed_dim=100):
    embeddings = []
    for seq in X_data:
        # Lấy trung bình của embeddings cho mỗi sequence
        seq_embeddings = []
        for token_id in seq:
            if token_id > 0:  # Bỏ qua padding
                # Tạo embedding ngẫu nhiên cho mỗi token (trong thực tế nên dùng pre-trained)
                embed = np.random.randn(embed_dim) * 0.1
                seq_embeddings.append(embed)
        if seq_embeddings:
            embeddings.append(np.mean(seq_embeddings, axis=0))
        else:
            embeddings.append(np.zeros(embed_dim))
    return np.array(embeddings)

start_time = time.time()
X_train_embed = create_embeddings(X_train)
X_test_embed = create_embeddings(X_test)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_embed, y_train_encoded)
xgb_time = time.time() - start_time

xgb_preds = xgb_model.predict(X_test_embed)
xgb_accuracy = accuracy_score(y_test_encoded, xgb_preds)
xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(y_test_encoded, xgb_preds, average='weighted')

models_results['XGBoost'] = {
    'accuracy': xgb_accuracy,
    'precision': xgb_precision,
    'recall': xgb_recall,
    'f1': xgb_f1,
    'time': xgb_time,
    'predictions': xgb_preds,
    'true_labels': y_test_encoded
}

# Lưu mô hình XGBoost
joblib.dump(xgb_model, '../models/xgb_model.joblib')
print(".4f")

# 5. Decision Tree
print("\n" + "="*50)
print("Training Decision Tree...")

start_time = time.time()
dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2
)

dt_model.fit(X_train, y_train_encoded)
dt_time = time.time() - start_time

dt_preds = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test_encoded, dt_preds)
dt_precision, dt_recall, dt_f1, _ = precision_recall_fscore_support(y_test_encoded, dt_preds, average='weighted')

models_results['Decision Tree'] = {
    'accuracy': dt_accuracy,
    'precision': dt_precision,
    'recall': dt_recall,
    'f1': dt_f1,
    'time': dt_time,
    'predictions': dt_preds,
    'true_labels': y_test_encoded
}

# Lưu mô hình Decision Tree
joblib.dump(dt_model, '../models/dt_model.joblib')
print(".4f")

# 6. Random Forest
print("\n" + "="*50)
print("Training Random Forest...")

start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

rf_model.fit(X_train, y_train_encoded)
rf_time = time.time() - start_time

rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test_encoded, rf_preds)
rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test_encoded, rf_preds, average='weighted')

models_results['Random Forest'] = {
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1': rf_f1,
    'time': rf_time,
    'predictions': rf_preds,
    'true_labels': y_test_encoded
}

# Lưu mô hình Random Forest
joblib.dump(rf_model, '../models/rf_model.joblib')
print(".4f")

# ==================== BERT (nếu có thể) ====================
try:
    print("\n" + "="*50)
    print("Trying BERT...")
    from transformers import BertTokenizer, BertForSequenceClassification
    from transformers import AdamW

    # Load preprocessed text data for BERT
    with open('../data/train_texts.txt', 'r', encoding='utf-8') as f:
        train_texts = f.read().split('\n')
    with open('../data/test_texts.txt', 'r', encoding='utf-8') as f:
        test_texts = f.read().split('\n')

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_classes)

    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(y_train_encoded)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(y_test_encoded)
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    start_time = time.time()
    model.to(device)
    model.train()

    for epoch in range(3):  # Ít epochs hơn cho BERT
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    bert_time = time.time() - start_time

    # Evaluate BERT
    model.eval()
    bert_preds = []
    bert_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            bert_preds.extend(preds.cpu().numpy())
            bert_labels.extend(labels.cpu().numpy())

    bert_accuracy = accuracy_score(bert_labels, bert_preds)
    bert_precision, bert_recall, bert_f1, _ = precision_recall_fscore_support(bert_labels, bert_preds, average='weighted')

    models_results['BERT'] = {
        'accuracy': bert_accuracy,
        'precision': bert_precision,
        'recall': bert_recall,
        'f1': bert_f1,
        'time': bert_time,
        'predictions': np.array(bert_preds),
        'true_labels': np.array(bert_labels)
    }

    # Lưu mô hình BERT
    model.save_pretrained('../models/bert_model')
    tokenizer.save_pretrained('../models/bert_tokenizer')
    print(".4f")

except Exception as e:
    print(f"BERT failed: {e}")
    print("Skipping BERT...")

# ==================== Tạo báo cáo ====================

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

# Tìm mô hình tốt nhất
best_model = max(models_results.items(), key=lambda x: x[1]['accuracy'])

print(f"Best Model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")

for model_name, results in models_results.items():
    print(f"\n{model_name}:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".2f")

# ==================== Tạo HTML report ====================

html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bước 3: Lựa Chọn Mô Hình - Báo Cáo Mở Rộng</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .metric-card.best {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            transform: scale(1.05);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        .best-row {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .code {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Bước 3: Lựa Chọn Mô Hình - Báo Cáo Mở Rộng</h1>

        <div class="highlight">
            <strong>📊 Tổng quan:</strong> So sánh 6 thuật toán học máy cho bài toán phân loại bệnh tật từ triệu chứng
        </div>

        <h2>📈 Kết Quả So Sánh Thuật Toán</h2>
        <div class="metrics-grid">
"""

# Thêm metrics cards cho tất cả 6 thuật toán
for i, (model_name, results) in enumerate(models_results.items()):
    best_class = "best" if model_name == best_model[0] else ""
    html_content += f"""
            <div class="metric-card {best_class}">
                <div class="metric-label">{model_name}</div>
                <div class="metric-value">{results['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
"""

html_content += """
        </div>

        <h2>📋 Bảng Chi Tiết Kết Quả</h2>
        <table>
            <thead>
                <tr>
                    <th>Thuật Toán</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Thời Gian (giây)</th>
                </tr>
            </thead>
            <tbody>
"""

for model_name, results in models_results.items():
    best_class = "best-row" if model_name == best_model[0] else ""
    html_content += f"""
                <tr class="{best_class}">
                    <td>{model_name}</td>
                    <td>{results['accuracy']:.4f}</td>
                    <td>{results['precision']:.4f}</td>
                    <td>{results['recall']:.4f}</td>
                    <td>{results['f1']:.4f}</td>
                    <td>{results['time']:.2f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>

        <h2>🏆 Thuật Toán Tốt Nhất</h2>
"""

best_name = best_model[0]
best_results = best_model[1]

html_content += f"""
        <div class="highlight">
            <strong>🥇 {best_name}</strong> đạt độ chính xác cao nhất với <strong>{best_results['accuracy']:.1%}</strong>
            <br>
            Thời gian huấn luyện: {best_results['time']:.2f} giây
        </div>

        <h2>📝 Báo Cáo Chi Tiết cho Thuật Toán Tốt Nhất ({best_name})</h2>
        <div class="code">
"""

# Classification report cho mô hình tốt nhất
from sklearn.metrics import classification_report
report = classification_report(best_results['true_labels'], best_results['predictions'], output_dict=False)
html_content += f"<pre>{report}</pre>"

html_content += """
        </div>

        <h2>🔍 Phân Tích Kết Quả</h2>
        <div class="highlight">
            <strong>Nhận xét:</strong><br>
            - Độ chính xác tổng thể: Thấp (dưới 50%) do đặc thù bài toán y khoa với triệu chứng chồng chéo<br>
            - Một số bệnh được phân loại tốt, một số bệnh khó phân biệt<br>
            - Cần cải thiện với dữ liệu lớn hơn và kỹ thuật preprocessing tốt hơn
        </div>

        <h2>💡 Khuyến Nghị</h2>
        <ul>
            <li>Tăng cường dữ liệu training với nhiều mẫu hơn</li>
            <li>Sử dụng kỹ thuật data augmentation cho text</li>
            <li>Thử nghiệm với pre-trained embeddings (Word2Vec, FastText)</li>
            <li>Áp dụng ensemble methods kết hợp nhiều mô hình</li>
            <li>Tune hyperparameters chi tiết hơn</li>
        </ul>

        <div class="footer">
            <p>⏰ Báo cáo được tạo vào: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>🧠 Hệ Chuyên Gia Bệnh Tật ViMedical - Bước 3: Lựa Chọn Mô Hình</p>
        </div>
    </div>
</body>
</html>
"""

with open('../reports/Step3_Model_Selection_Expanded.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ Báo cáo HTML đã được tạo: ../reports/Step3_Model_Selection_Expanded.html")

# Lưu kết quả tổng hợp
results_summary = {
    'models': models_results,
    'best_model': best_model[0],
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

joblib.dump(results_summary, '../models/step3_results_expanded.joblib')

print("✅ Đã lưu kết quả tổng hợp vào: ../models/step3_results_expanded.joblib")
print("\n🎉 Hoàn thành Bước 3: Lựa Chọn Mô Hình (mở rộng)!")