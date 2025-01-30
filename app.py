from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

app = Flask(__name__)

# Load or train the anomaly detection model
MODEL_PATH = 'model/isolation_forest_model.pkl'

def train_model():
    # Sample training data - replace with actual medical IoT data
    data = pd.DataFrame({
        'temperature': np.random.normal(37, 0.5, 1000),
        'heart_rate': np.random.normal(70, 5, 1000),
        'oxygen_level': np.random.normal(98, 1, 1000)
    })
    model = IsolationForest(contamination=0.01)
    model.fit(data)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            heart_rate = float(request.form['heart_rate'])
            oxygen_level = float(request.form['oxygen_level'])
            input_data = pd.DataFrame([[temperature, heart_rate, oxygen_level]],
                                      columns=['temperature', 'heart_rate', 'oxygen_level'])
            prediction = model.predict(input_data)
            if prediction[0] == -1:
                result = 'Anomaly detected! Please consult a healthcare professional.'
            else:
                result = 'No anomalies detected. Parameters are within normal ranges.'
        except ValueError:
            result = 'Invalid input. Please enter valid numerical values.'
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
