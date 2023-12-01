# TC.py
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('your_model_filename.pkl')
scaler = joblib.load('your_scaler_filename.pkl')

def process_text_feature(feature_text):
    # Thực hiện logic mã hóa dữ liệu chữ của bạn ở đây
    # Ví dụ: Chỉ là một biến bổ sung, bạn có thể cần thêm mã hóa One-Hot, Vectorize Text, hoặc sử dụng TF-IDF, vvv.
    # Đây chỉ là ví dụ đơn giản, bạn cần thay thế nó bằng logic thực tế của bạn.
    return len(feature_text)  # Ví dụ: Trả về độ dài của feature_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_text = request.form.get('feature_text')

    try:
        # Chuẩn bị dữ liệu cho dự đoán (sử dụng logic mã hóa dữ liệu chữ)
        input_data = process_text_feature(feature_text)

        # Chuẩn hóa dữ liệu
        input_data_scaled = scaler.transform([[input_data]])

        # Dự đoán
        prediction = model.predict(input_data_scaled)

        return render_template('index.html', prediction=f"Predicted class: {prediction[0]}")
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
