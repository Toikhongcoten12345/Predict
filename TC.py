from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('your_model_filename.pkl')
scaler = joblib.load('your_scaler_filename.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature1 = request.form.get('feature1')
    feature2 = request.form.get('feature2')

    try:
        feature1 = float(feature1)
        feature2 = float(feature2)
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please provide numeric values.")

    # Tiếp tục với code của bạn...

if __name__ == '__main__':
    app.run(debug=True)
