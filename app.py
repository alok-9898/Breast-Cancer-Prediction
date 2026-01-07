from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle

try:
    model = pickle.load(open("models/model.pkl", "rb"))
except Exception as _e:
    model = None

# Flask app (templates live in `template/` folder)
app = Flask(__name__, template_folder='template', static_folder='static')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', message=["Model not available"])

    features = request.form.get('feature', '')
    if not features:
        return render_template('index.html', message=["Please provide comma-separated numeric features"]) 

    features_list = [f.strip() for f in features.split(',') if f.strip()]
    try:
        np_features = np.array(features_list, dtype=np.float32)
    except ValueError:
        return render_template('index.html', message=["Invalid input: ensure all values are numeric"]) 

    try:
        prediction = model.predict(np_features.reshape(1, -1))
    except Exception as e:
        return render_template('index.html', message=[f"Prediction error: {e}"])

    output = ["CANCEROUS" if int(prediction[0]) == 1 else "NON-CANCEROUS"]
    return render_template('index.html', message=output)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if model is None:
        return jsonify({'error': 'Model not available'}), 503

    data = request.get_json(silent=True) or {}
    features = data.get('feature') or data.get('features') or ''
    if not features:
        return jsonify({'error': 'Missing feature value (provide comma-separated numbers)'}), 400

    features_list = [f.strip() for f in str(features).split(',') if f.strip()]
    try:
        np_features = np.array(features_list, dtype=np.float32)
    except ValueError:
        return jsonify({'error': 'Invalid input: ensure all values are numeric'}), 400

    try:
        prediction = model.predict(np_features.reshape(1, -1))
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

    result = 'CANCEROUS' if int(prediction[0]) == 1 else 'NON-CANCEROUS'
    return jsonify({'prediction': result})

#main driver function
if __name__ == "__main__":
    app.run(debug=True)