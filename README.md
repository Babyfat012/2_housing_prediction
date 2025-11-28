# 🏡 Housing Price Prediction

Dự án mini về Machine Learning để dự đoán giá nhà sử dụng California Housing Dataset.

## 📋 Mục đích

Dự án này được tạo ra với mục đích học tập về:
- Machine Learning cơ bản
- Tiền xử lý dữ liệu (Data Preprocessing)
- Huấn luyện và đánh giá mô hình
- Deploy ứng dụng ML đơn giản với Streamlit

## 🎯 Tổng quan

Dự án sử dụng **Linear Regression** để dự đoán giá nhà trung bình ở California dựa trên các đặc điểm như:
- Vị trí địa lý (longitude, latitude)
- Tuổi của nhà
- Số phòng
- Dân số khu vực
- Thu nhập trung bình
- Khoảng cách tới đại dương

## 📁 Cấu trúc dự án

```
Housing_Prediction/
│
├── housing_prediction.ipynb    # Notebook chính chứa toàn bộ pipeline
├── app.py                       # Ứng dụng Streamlit (được tạo từ notebook)
├── house_price_model.pkl        # Mô hình đã được huấn luyện
└── README.md                    # File tài liệu này
```

## 🔧 Công nghệ sử dụng

### Thư viện Python
- **pandas**: Xử lý và phân tích dữ liệu
- **numpy**: Tính toán số học
- **matplotlib**: Trực quan hóa dữ liệu
- **scikit-learn**: Thư viện Machine Learning
  - `LinearRegression`: Mô hình hồi quy tuyến tính
  - `RandomForestRegressor`: Mô hình rừng ngẫu nhiên (đã import)
  - Các metrics đánh giá: MAE, MSE, R², MAPE
- **joblib**: Lưu và tải mô hình
- **streamlit**: Tạo web app
- **pyngrok**: Tạo public URL cho ứng dụng

## 📊 Dataset

**California Housing Dataset**
- Nguồn: [Hands-On Machine Learning GitHub](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)
- Số lượng mẫu: ~20,640 mẫu
- Số đặc trưng: 10 cột

### Các đặc trưng (Features):
1. `longitude`: Kinh độ
2. `latitude`: Vĩ độ
3. `housing_median_age`: Tuổi trung bình của nhà
4. `total_rooms`: Tổng số phòng
5. `total_bedrooms`: Tổng số phòng ngủ
6. `population`: Dân số
7. `households`: Số hộ gia đình
8. `median_income`: Thu nhập trung bình
9. `ocean_proximity`: Khoảng cách tới đại dương (categorical)

### Biến mục tiêu (Target):
- `median_house_value`: Giá trị trung bình của nhà

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường

```bash
pip install pandas numpy matplotlib scikit-learn joblib streamlit pyngrok
```

### 2. Chạy Notebook

Mở file `housing_prediction.ipynb` trong Jupyter Notebook hoặc JupyterLab và chạy từng cell theo thứ tự.

### 3. Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Ứng dụng sẽ chạy tại `http://localhost:8501`

## 📈 Pipeline Machine Learning

### 1. **Tải dữ liệu**
```python
data = pd.read_csv(url)
```

### 2. **Tiền xử lý**
- Loại bỏ giá trị null: `data.dropna()`
- One-Hot Encoding cho biến `ocean_proximity`

### 3. **Chia dữ liệu**
- **Training set**: 80%
- **Test set**: 20%
- `random_state=42` để tái tạo kết quả

### 4. **Huấn luyện mô hình**
```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

### 5. **Đánh giá mô hình**

Các chỉ số đánh giá:
- **MAE** (Mean Absolute Error): Sai số trung bình tuyệt đối
- **MSE** (Mean Squared Error): Sai số bình phương trung bình
- **R² Score**: Độ chính xác của mô hình (0-1, càng cao càng tốt)
- **MAPE** (Mean Absolute Percentage Error): Sai số phần trăm

### 6. **Trực quan hóa**
- Scatter plot so sánh giá trị thực tế vs dự đoán
- Đường chéo 45 độ thể hiện dự đoán hoàn hảo

### 7. **Lưu mô hình**
```python
joblib.dump(lr_model, "house_price_model.pkl")
```

## 🎨 Ứng dụng Streamlit

Ứng dụng web cho phép người dùng:
- Nhập các thông tin về căn nhà
- Nhấn nút "Dự đoán giá nhà"
- Xem kết quả dự đoán giá nhà

### Giao diện
- **Sidebar**: Form nhập liệu với các trường:
  - Các giá trị số: longitude, latitude, age, rooms, etc.
  - Checkbox cho ocean_proximity
- **Main panel**: Hiển thị kết quả dự đoán

## 📚 Kiến thức học được

### 1. **Data Preprocessing**
- Xử lý missing values
- One-Hot Encoding cho biến categorical
- Chia train/test set

### 2. **Machine Learning**
- Linear Regression
- Training và prediction
- Model evaluation metrics

### 3. **Data Visualization**
- Matplotlib scatter plots
- So sánh actual vs predicted values

### 4. **Model Deployment**
- Lưu model với joblib
- Tạo web app với Streamlit
- Sử dụng ngrok để public app

## 🔍 Cải tiến có thể thực hiện

1. **Feature Engineering**
   - Tạo thêm features mới (rooms per household, population per household)
   - Scaling/Normalization

2. **Thử các mô hình khác**
   - Random Forest (đã import)
   - Gradient Boosting
   - Neural Networks

3. **Hyperparameter Tuning**
   - Grid Search
   - Random Search

4. **Cross-validation**
   - K-Fold Cross Validation

5. **Cải thiện UI**
   - Thêm biểu đồ vào Streamlit app
   - Hiển thị model metrics
   - Input validation

## 📝 Ghi chú

- Mô hình Linear Regression là mô hình đơn giản nhất, phù hợp cho việc học tập
- Dataset California Housing là dataset phổ biến trong các khóa học ML
- R² score thường dao động từ 0.6-0.7 với Linear Regression trên dataset này

## 📧 Liên hệ

Dự án này được tạo ra cho mục đích học tập về AI và Machine Learning.

## 📄 License

MIT License - Free to use for educational purposes

---

**Happy Learning! 🎓**

