from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("models/model.pkl","rb"))

#Flask app

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = request.form['feature']
    features_list = features.split(',')
    np_features = np.array(features_list, dtype=np.float32)
    prediction = model.predict(np_features.reshape(1,-1))

    output = ["CANCEROUS" if prediction[0] == 1 else "NON-CANCEROUS"]
    return render_template('index.html', message = output)

#main driver function
if __name__ == "__main__":
    app.run(debug=True)