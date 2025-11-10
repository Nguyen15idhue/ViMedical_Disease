# Quy Trình Xây Dựng Hệ Chuyên Gia Chẩn Đoán Đơn Giản
## Dựa trên Bộ Dữ Liệu ViMedical_Disease cho Môn Nhập Môn Trí Tuệ Nhân Tạo

**Tác giả**: Sinh viên (dựa trên hướng dẫn AI)  
**Ngày**: 10/11/2025  
**Mục tiêu**: Xây dựng hệ chuyên gia chẩn đoán bệnh dựa trên triệu chứng, sử dụng phương pháp học máy với PyTorch và GPU. Tài liệu này dành cho môn Nhập môn Trí Tuệ Nhân Tạo, tập trung giải thích khái niệm cơ bản, bản chất, và quy trình thực tế.

---

## Giới Thiệu
Hệ chuyên gia là một chương trình máy tính mô phỏng chuyên gia con người trong một lĩnh vực cụ thể (ở đây là chẩn đoán y tế sơ bộ dựa trên triệu chứng). Chúng ta sử dụng bộ dữ liệu ViMedical_Disease (CSV với 23.522 dòng, 2 cột: Disease và Question) để biểu diễn tri thức "triệu chứng → bệnh". Phương pháp chính: So sánh 3 thuật toán tối ưu (Random Forest, SVM, XGBoost) để chọn mô hình tốt nhất, triển khai với PyTorch trên GPU.

**Tại sao phù hợp với môn AI?** Đây là ví dụ thực tế về biểu diễn tri thức, suy luận, và học máy giám sát (supervised learning), giúp hiểu cách máy tính "học" và "suy nghĩ" như con người.

**Công cụ cần**: Python, PyTorch (cho deep learning và GPU), scikit-learn (cho baseline), pandas (xử lý dữ liệu), underthesea (NLP tiếng Việt), CUDA (cho GPU nếu có).

---

## Quy Trình Chi Tiết

### Bước 1: Thu Thập và Chuẩn Bị Dữ Liệu (Data Collection & Preparation)
**Sẽ làm gì?** Thu thập và tổ chức dữ liệu từ ViMedical_Disease.csv để làm nền tảng tri thức cho hệ thống.

**Làm như nào?**
- Tải và đọc file CSV bằng pandas.
- Tách dữ liệu thành tập huấn luyện (training set: 80%) và tập kiểm tra (test set: 20%) bằng train_test_split từ scikit-learn. Ví dụ: `train, test = train_test_split(data, test_size=0.2, stratify=data['Disease'])`.
- Lưu trữ dữ liệu đã tách thành file train_data.csv và test_data.csv.

**Ở bước này cần nắm được thông tin gì?**
- Cấu trúc dataset: 603 bệnh, mỗi bệnh có ~20 câu hỏi triệu chứng.
- Số lượng dữ liệu: 12.060 dòng sau xử lý, đủ cho mô hình.

**Khái niệm nào?**
- **Tri thức (Knowledge)**: Thông tin mà hệ thống biết, ở đây là mối liên hệ triệu chứng-bệnh từ dữ liệu thực tế.
- **Tập huấn luyện (Training Set)**: Dữ liệu dùng để "dạy" mô hình.
- **Tập kiểm tra (Test Set)**: Dữ liệu dùng để đánh giá mô hình sau khi train.

**Bản chất gì?** Dữ liệu là "nguồn tri thức" thô; bước này chuyển nó thành dạng có thể sử dụng, đảm bảo mô hình học từ đa dạng dữ liệu (tránh bias nếu chỉ dùng một phần).

**Làm xong bước này hiểu gì?** Hiểu cách chuẩn bị dữ liệu cho AI: Dữ liệu là nền tảng, và tách train/test giúp đánh giá công bằng.

---

### Bước 2: Xử Lý Dữ Liệu (Data Preprocessing)
**Sẽ làm gì?** Chuyển dữ liệu văn bản tiếng Việt thành dạng số mà máy tính hiểu được.

**Làm như nào?**
- Trích xuất triệu chứng từ cột Question: Sử dụng underthesea để tách từ, loại bỏ từ dừng (stopwords như "tôi", "đang"). Ví dụ: "Tôi đang cảm thấy mệt mỏi" → ["mệt mỏi"].
- Tạo vector nhị phân: Mỗi triệu chứng là một cột (1 nếu có, 0 nếu không). Sử dụng CountVectorizer từ scikit-learn.
- Xử lý nhãn: Cột Disease là nhãn (label) để dự đoán, chuyển thành số bằng LabelEncoder.

**Ở bước này cần nắm được thông tin gì?**
- Các triệu chứng phổ biến (ví dụ: "mệt mỏi", "chóng mặt" cho bệnh tim mạch).
- Độ dài vector: 1.000 features, vocabulary size.

**Khái niệm nào?**
- **Biểu diễn tri thức (Knowledge Representation)**: Cách lưu trữ tri thức để máy tính xử lý (từ văn bản sang số).
- **Vector hóa (Vectorization)**: Biến văn bản thành vector số.
- **Xử lý ngôn ngữ tự nhiên (NLP)**: Kỹ thuật giúp máy hiểu ngôn ngữ con người.

**Bản chất gì?** Văn bản là "tri thức con người"; vector là "tri thức máy". Bước này loại bỏ nhiễu (từ thừa) để mô hình tập trung vào triệu chứng quan trọng.

**Làm xong bước này hiểu gì?** Hiểu NLP cơ bản trong AI: Máy không hiểu văn bản trực tiếp, cần chuyển đổi để "học" quy luật.

---

### Bước 3: Chọn Phương Pháp Biểu Diễn Tri Thức và Xây Dựng Mô Hình (Model Selection & Building)
**Sẽ làm gì?** Chọn và so sánh 3 thuật toán tối ưu (Random Forest, SVM, XGBoost) để biểu diễn tri thức, triển khai với PyTorch trên GPU.

**Làm như nào?**
- Chọn 3 thuật toán: RandomForestClassifier (scikit-learn), SVC (SVM), XGBClassifier (XGBoost).
- Triển khai với PyTorch: Chuyển dữ liệu thành tensor, dùng GPU nếu có (torch.cuda.is_available()).
- Khởi tạo mô hình: Ví dụ, Random Forest với n_estimators=100, SVM với kernel='linear', XGBoost với n_estimators=100.
- Chuẩn bị input: Vector triệu chứng (X) và nhãn bệnh (y) dưới dạng tensor.

**Ở bước này cần nắm được thông tin gì?**
- Ưu điểm từng thuật toán: Random Forest dễ dùng, SVM cho text, XGBoost cho accuracy cao.
- Tham số: n_estimators, kernel, learning_rate.
- GPU: Tăng tốc train nếu dataset lớn.

**Khái niệm nào?**
- **Random Forest**: Ensemble của nhiều cây quyết định, giảm overfitting.
- **SVM (Support Vector Machine)**: Tìm hyperplane tối ưu để phân tách class.
- **XGBoost**: Gradient boosting, cải thiện lỗi tuần tự.
- **PyTorch**: Framework deep learning, hỗ trợ GPU cho tính toán nhanh.

**Bản chất gì?** Các thuật toán biểu diễn tri thức như quy tắc hoặc xác suất từ dữ liệu. PyTorch cho phép tính toán song song trên GPU, tăng hiệu quả.

**Làm xong bước này hiểu gì?** Hiểu biểu diễn tri thức trong AI: Chọn thuật toán phù hợp dữ liệu, và lợi ích của GPU trong train.

---

### Bước 4: Huấn luyện Mô Hình (Model Training)
**Sẽ làm gì?** "Dạy" 3 mô hình học quy luật từ dữ liệu huấn luyện, dùng PyTorch và GPU.

**Làm như nào?**
- Fit mô hình: `model.fit(X_train, y_train)` cho mỗi thuật toán, với dữ liệu trên GPU nếu có.
- PyTorch: Chuyển mô hình và dữ liệu sang device (cuda nếu GPU).
- Lưu mô hình đã train để sử dụng sau.

**Ở bước này cần nắm được thông tin gì?**
- Thời gian train: Nhanh hơn với GPU (vài phút thay vì giờ).
- Kết quả train: Loss giảm dần.

**Khái niệm nào?**
- **Huấn luyện (Training)**: Quá trình tối ưu hóa mô hình để giảm lỗi dự đoán.
- **GPU (Graphics Processing Unit)**: Bộ xử lý đồ họa, tăng tốc tính toán ma trận trong AI.
- **Overfitting**: Mô hình quá khớp dữ liệu train, kém trên dữ liệu mới.

**Bản chất gì?** Train là "học": Mô hình tìm quy luật từ dữ liệu, GPU giúp xử lý nhanh dữ liệu lớn.

**Làm xong bước này hiểu gì?** Hiểu học máy: Máy "học" bằng tối ưu, GPU làm cho quy trình hiệu quả hơn.

---

### Bước 5: Đánh Giá Mô Hình (Model Evaluation)
**Sẽ làm gì?** Kiểm tra và so sánh độ chính xác của 3 mô hình trên dữ liệu test.

**Làm như nào?**
- Dự đoán trên test set: `predictions = model.predict(X_test)`.
- Tính metrics: Accuracy, Precision, Recall, F1-Score cho mỗi mô hình.
- So sánh: Bảng kết quả, chọn mô hình tốt nhất (Random Forest thường cao nhất).

**Ở bước này cần nắm được thông tin gì?**
- Độ chính xác: Mong đợi 40-50% (thấp vì triệu chứng chồng chéo).
- Lỗi phổ biến: Bệnh có triệu chứng giống nhau.

**Khái niệm nào?**
- **Đánh giá (Evaluation)**: Đo lường hiệu quả mô hình.
- **Confusion Matrix**: Bảng đúng/sai.
- **Precision/Recall**: Đo lường sai sót.

**Bản chất gì?** Đánh giá kiểm tra "thông minh" thực sự, so sánh thuật toán để chọn tối ưu.

**Làm xong bước này hiểu gì?** Hiểu validation trong AI: Đánh giá khách quan, chọn mô hình dựa trên metrics.

---

### Bước 6: Triển Khai Hệ Thống (Deployment & Inference)
**Sẽ làm gì?** Tạo cơ chế suy luận và giao diện, dùng mô hình tốt nhất.

**Làm như nào?**
- Xây dựng hàm suy luận: Nhận triệu chứng, chạy predict, trả về bệnh + giải thích.
- Giao diện: Streamlit hoặc HTML với form nhập triệu chứng.
- Demo: Chạy với ví dụ.

**Ở bước này cần nắm được thông tin gì?**
- Đầu vào: Triệu chứng text.
- Đầu ra: Bệnh + lý do.

**Khái niệm nào?**
- **Suy luận (Reasoning/Inference)**: Áp dụng mô hình.
- **Giải thích (Explainability)**: Hiển thị lý do.
- **Giao diện người dùng (User Interface)**: Tương tác.

**Bản chất gì?** Suy luận áp dụng tri thức học được, giao diện làm cho hệ dễ dùng.

**Làm xong bước này hiểu gì?** Hiểu hệ chuyên gia hoàn chỉnh: Từ dữ liệu đến ứng dụng thực tế.

---

## Kết Luận: Làm Xong Tất Cả Cần Hiểu Gì?
Sau khi hoàn thành quy trình, bạn sẽ hiểu:
- **Bản chất AI trong hệ chuyên gia**: AI biểu diễn tri thức, học quy luật, áp dụng suy luận.
- **Quy trình học máy**: Từ dữ liệu → xử lý → mô hình → train → đánh giá → triển khai.
- **Ứng dụng thực tế**: Hệ hỗ trợ chẩn đoán, nhưng cần chuyên gia.
- **Bài học cho môn AI**: Khái niệm cốt lõi như tri thức, biểu diễn, suy luận.

---

## Phần Bổ Sung: Trả Lời Các Câu Hỏi Thường Gặp Từ Giáo Viên

### Pipeline Tổng Thể Của Dự Án
Pipeline: Input Dữ Liệu → Xử Lý Dữ Liệu → Xây Dựng Mô Hình (3 thuật toán) → Train (PyTorch/GPU) → Đánh Giá → Triển Khai.

### Thuật Toán Sử Dụng
3 thuật toán tối ưu: Random Forest, SVM, XGBoost. Chọn dựa trên accuracy, giải thích.

### Dữ Liệu Là Gì?
ViMedical_Disease.csv: 23.522 dòng, 603 bệnh, text tiếng Việt.

### Train/Test Và Kết Quả
Train: 80%, Test: 20%. Kết quả: Random Forest ~40%, SVM ~35%, XGBoost ~45%.

### Biểu Đồ Và Giải Thích
Confusion Matrix, Tree Plot, Accuracy vs. Params. Giải thích: Hiển thị lỗi, quy luật.

---

## So Sánh 3 Thuật Toán Train

### Kết Quả So Sánh (Từ Chạy Thực Tế)
| Thuật Toán      | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| Decision Tree  | 0.83%   | 0.81%    | 0.83% | 0.81%   |
| Naive Bayes    | 28.52%  | 33.44%   | 28.52%| 25.80%  |
| Random Forest  | 40.51%  | 42.54%   | 40.51%| 39.00%  |

### 3 Thuật Toán Tối Ưu Nhất
1. **Random Forest**: Tối ưu tổng thể, accuracy cao.
2. **SVM**: Cho text, robust.
3. **XGBoost**: Performance tối đa.

**Khuyến nghị**: Dùng Random Forest, triển khai với PyTorch/GPU.

---

**Lưu ý**: Sử dụng GPU với PyTorch để tăng tốc. Tham khảo: PyTorch docs, scikit-learn docs.