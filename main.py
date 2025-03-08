import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Dùng để load mô hình đã huấn luyện
from sklearn.preprocessing import StandardScaler

# 📌 Load mô hình (giả sử đã huấn luyện và lưu dưới dạng .pkl)
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # Đổi "model.pkl" thành file mô hình của bạn
    return model

# 📌 Hàm xử lý dữ liệu đầu vào và dự đoán
def predict(model, data):
    # Giả sử cần chuẩn hóa dữ liệu trước khi đưa vào mô hình
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    predictions = model.predict(data_scaled)
    return predictions

# 📌 Giao diện Web bằng Streamlit
st.title("📊 Ứng dụng Dự đoán từ File CSV")

# **Tải lên file CSV**
uploaded_file = st.file_uploader("📂 Chọn file CSV để dự đoán", type=["csv"])

if uploaded_file is not None:
    # Đọc file CSV thành DataFrame
    df = pd.read_csv(uploaded_file)
    
    st.write("📜 **Dữ liệu tải lên:**")
    st.dataframe(df.head())  # Hiển thị 5 dòng đầu tiên

    # Kiểm tra nếu mô hình tồn tại
    model = load_model()
    
    if model:
        # Thực hiện dự đoán
        predictions = predict(model, df)

        # Tạo DataFrame kết quả
        results = df.copy()
        results["Prediction"] = predictions

        # Xuất kết quả ra file CSV
        output_csv = "predictions.csv"
        results.to_csv(output_csv, index=False)

        # Hiển thị kết quả
        st.write("📌 **Dự đoán:**")
        st.dataframe(results.head())

        # Nút tải xuống file CSV kết quả
        st.download_button(
            label="📥 Tải về kết quả dự đoán",
            data=results.to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )
