import pandas as pd
import numpy as np
import joblib
import time
from collections import Counter
from underthesea import word_tokenize

# Load dữ liệu gốc
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Tạo lại cột Processed_Question để demo
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # Remove stopwords (simple list)
    stopwords = ['tôi', 'đang', 'là', 'có', 'không', 'và', 'hoặc', 'nhưng', 'nếu', 'thì']
    tokens = [word for word in tokens if word not in stopwords and word.isalpha()]
    return ' '.join(tokens)

# Apply preprocessing
train_data['Processed_Question'] = train_data['Question'].apply(preprocess_text)

# Load dữ liệu đã xử lý
X_train_ml = np.load('data/X_train.npy')
X_test_ml = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

# Load deep learning data
X_train_dl = np.load('data/X_train_dl.npy')
X_test_dl = np.load('data/X_test_dl.npy')

# Load tokenizer và metadata
tokenizer = joblib.load('data/tokenizer.joblib')
label_encoder = joblib.load('data/label_encoder.joblib')
dl_metadata = joblib.load('data/dl_metadata.joblib')

# Tạo HTML report
html_content = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ViMedical - Bước 2: Tiền Xử Lý Dữ Liệu</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            color: #6c757d;
            margin-top: 5px;
        }
        .highlight {
            background: #e8f4fd;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }
        .code {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            color: #6c757d;
        }
        .success {
            color: #28a745;
            font-weight: bold;
        }
        .info {
            color: #17a2b8;
        }
        .warning {
            color: #ffc107;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 ViMedical - Hệ Chuyên Gia Bệnh Tật</h1>
        <h2>📊 Bước 2: Tiền Xử Lý Dữ Liệu</h2>

        <div class="highlight">
            <strong>🎯 Mục tiêu:</strong> Chuẩn bị và tiền xử lý dữ liệu văn bản tiếng Việt cho các mô hình học máy và học sâu
        </div>

        <h2>📈 Tổng Quan Dữ Liệu</h2>
        <div class="metric-grid">
"""

# Thêm metrics cho dữ liệu gốc
html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{len(train_data)}</div>
                <div class="metric-label">Mẫu Training</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(test_data)}</div>
                <div class="metric-label">Mẫu Test</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(np.unique(y_train))}</div>
                <div class="metric-label">Số Lớp Bệnh Tật</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(train_data['Question'].str.split().sum())/len(train_data):.1f}</div>
                <div class="metric-label">Từ Trung Bình/Câu</div>
            </div>
"""

html_content += """
        </div>

        <h2>🔤 Phân Tích Văn Bản Gốc</h2>
        <table>
            <thead>
                <tr>
                    <th>Thống Kê</th>
                    <th>Giá Trị</th>
                </tr>
            </thead>
            <tbody>
"""

# Phân tích văn bản
questions = train_data['Question']
word_counts = questions.str.split().str.len()
char_counts = questions.str.len()

html_content += f"""
                <tr>
                    <td>Tổng số câu hỏi</td>
                    <td>{len(questions)}</td>
                </tr>
                <tr>
                    <td>Số từ trung bình/câu</td>
                    <td>{word_counts.mean():.1f}</td>
                </tr>
                <tr>
                    <td>Số từ tối đa/câu</td>
                    <td>{word_counts.max()}</td>
                </tr>
                <tr>
                    <td>Số từ tối thiểu/câu</td>
                    <td>{word_counts.min()}</td>
                </tr>
                <tr>
                    <td>Số ký tự trung bình/câu</td>
                    <td>{char_counts.mean():.1f}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>

        <h2>🧹 Tiền Xử Lý Văn Bản</h2>
        <div class="highlight">
            <strong>Các bước xử lý:</strong>
            <ul>
                <li>Chuyển về chữ thường</li>
                <li>Tokenization bằng Underthesea</li>
                <li>Loại bỏ stopwords cơ bản</li>
                <li>Giữ lại chỉ từ alphabetic</li>
            </ul>
        </div>

        <h3>📝 Ví Dụ Văn Bản Đã Xử Lý</h3>
        <table>
            <thead>
                <tr>
                    <th>Văn Bản Gốc</th>
                    <th>Văn Bản Đã Xử Lý</th>
                </tr>
            </thead>
            <tbody>
"""

# Hiển thị 5 ví dụ đầu tiên
for i in range(min(5, len(train_data))):
    original = train_data['Question'].iloc[i][:100] + "..." if len(train_data['Question'].iloc[i]) > 100 else train_data['Question'].iloc[i]
    processed = train_data['Processed_Question'].iloc[i][:100] + "..." if len(str(train_data['Processed_Question'].iloc[i])) > 100 else str(train_data['Processed_Question'].iloc[i])
    html_content += f"""
                <tr>
                    <td>{original}</td>
                    <td>{processed}</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>

        <h2>🤖 Chuẩn Bị Dữ Liệu Cho Các Mô Hình</h2>

        <h3>📊 Dữ Liệu Cho Mô Hình Học Máy Truyền Thống</h3>
        <div class="metric-grid">
"""

html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{X_train_ml.shape[1]}</div>
                <div class="metric-label">Kích Thước Từ Vựng</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{X_train_ml.shape[0]}</div>
                <div class="metric-label">Số Mẫu Training</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{X_train_ml.shape[1]}</div>
                <div class="metric-label">Số Tính Năng</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{np.sum(X_train_ml) / X_train_ml.size:.3f}</div>
                <div class="metric-label">Tỷ Lệ Sparsity</div>
            </div>
"""

html_content += """
        </div>

        <h3>🧠 Dữ Liệu Cho Mô Hình Học Sâu</h3>
        <div class="metric-grid">
"""

vocab_size = dl_metadata['vocab_size']
max_seq_len = dl_metadata['max_sequence_length']

html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{vocab_size}</div>
                <div class="metric-label">Kích Thước Từ Vựng</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{X_train_dl.shape[0]}</div>
                <div class="metric-label">Số Mẫu Training</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{max_seq_len}</div>
                <div class="metric-label">Độ Dài Sequence Tối Đa</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(tokenizer.word_index)}</div>
                <div class="metric-label">Tổng Số Từ Duy Nhất</div>
            </div>
"""

html_content += """
        </div>

        <h3>🔢 Ví Dụ Tokenization</h3>
        <div class="code">
"""

# Hiển thị ví dụ tokenization
sample_text = train_data['Processed_Question'].iloc[0]
sample_sequence = X_train_dl[0][:20]  # Chỉ hiển thị 20 token đầu

html_content += f"""
<strong>Văn bản gốc:</strong> {sample_text}<br><br>
<strong>Sequence (20 token đầu):</strong> {sample_sequence.tolist()}<br><br>
<strong>Mapping một số token:</strong><br>
"""

# Hiển thị mapping cho một số token phổ biến
word_index = tokenizer.word_index
for word, idx in list(word_index.items())[:10]:
    html_content += f"{word} → {idx}<br>"

html_content += """
        </div>

        <h2>🏷️ Phân Tích Nhãn (Labels)</h2>
        <div class="metric-grid">
"""

html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{len(label_encoder.classes_)}</div>
                <div class="metric-label">Tổng Số Lớp Bệnh</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{max(Counter(y_train).values())}</div>
                <div class="metric-label">Lớp Có Nhiều Mẫu Nhất</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{min(Counter(y_train).values())}</div>
                <div class="metric-label">Lớp Có Ít Mẫu Nhất</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{Counter(y_train).most_common(1)[0][0]}</div>
                <div class="metric-label">ID Lớp Phổ Biến Nhất</div>
            </div>
"""

html_content += """
        </div>

        <h3>📋 Top 10 Lớp Bệnh Phổ Biến Nhất</h3>
        <table>
            <thead>
                <tr>
                    <th>Tên Bệnh</th>
                    <th>Số Mẫu</th>
                    <th>Tỷ Lệ (%)</th>
                </tr>
            </thead>
            <tbody>
"""

# Top 10 classes
label_counts = Counter(y_train)
total_samples = len(y_train)
for label_id, count in label_counts.most_common(10):
    disease_name = label_encoder.inverse_transform([label_id])[0]
    percentage = (count / total_samples) * 100
    html_content += f"""
                <tr>
                    <td>{disease_name}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
"""

html_content += """
            </tbody>
        </table>

        <h2>💾 Tệp Đã Lưu</h2>
        <div class="highlight">
            <strong>✅ Dữ liệu đã được lưu thành công:</strong>
            <ul>
                <li><code>data/X_train.npy</code> - Features training (Traditional ML)</li>
                <li><code>data/y_train.npy</code> - Labels training</li>
                <li><code>data/X_test.npy</code> - Features test (Traditional ML)</li>
                <li><code>data/y_test.npy</code> - Labels test</li>
                <li><code>data/X_train_dl.npy</code> - Sequences training (Deep Learning)</li>
                <li><code>data/X_test_dl.npy</code> - Sequences test (Deep Learning)</li>
                <li><code>data/tokenizer.joblib</code> - Tokenizer cho Deep Learning</li>
                <li><code>data/label_encoder.joblib</code> - Label Encoder</li>
                <li><code>data/dl_metadata.joblib</code> - Metadata cho Deep Learning</li>
            </ul>
        </div>

        <h2>📝 Tổng Kết</h2>
        <div class="highlight">
            <strong>🎯 Đã hoàn thành tiền xử lý dữ liệu:</strong><br>
            - Xử lý <strong>{len(train_data)}</strong> mẫu training và <strong>{len(test_data)}</strong> mẫu test<br>
            - Tạo dữ liệu cho <strong>{len(np.unique(y_train))}</strong> loại bệnh khác nhau<br>
            - Chuẩn bị dữ liệu cho cả mô hình học máy truyền thống và học sâu<br>
            - Từ vựng: <strong>{vocab_size}</strong> từ cho Deep Learning, <strong>{X_train_ml.shape[1]}</strong> tính năng cho Traditional ML<br>
            - Sẵn sàng cho bước tiếp theo: Lựa chọn và huấn luyện mô hình
        </div>

        <div class="footer">
            <p>⏰ Báo cáo được tạo vào: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>🧠 Hệ Chuyên Gia Bệnh Tật ViMedical - Bước 2: Tiền Xử Lý Dữ Liệu</p>
        </div>
    </div>
</body>
</html>
"""

# Lưu file HTML
with open('reports/Step2_Data_Preprocessing.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ Đã tạo báo cáo HTML: reports/Step2_Data_Preprocessing.html")