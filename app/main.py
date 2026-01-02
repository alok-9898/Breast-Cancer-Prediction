from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl","rb"))

#Flask app

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    pass






#main driver function
if __name___ == "__main__":
    app.run(debug=True)