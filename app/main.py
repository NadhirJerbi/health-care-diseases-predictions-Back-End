from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import pickle
import os
import json

app = Flask(__name__)

cros =CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

basedir = os.path.abspath(os.path.dirname(__file__))
heartmodelfile = os.path.join(basedir, './pkl/heart_model.pickle')
heartModel = pickle.load(open(heartmodelfile, 'rb'))

@app.route("/" )
@cross_origin()
def home_view():
        print('welcome')
        return jsonify(str("Class "))

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        
        to_predict_list = request.json
        
        to_predict_list = list(to_predict_list.values())
      
        to_predict_list = list(map(float, to_predict_list))

        to_predict = np.array(to_predict_list).reshape(1,7)

        result = heartModel.predict(to_predict)

        prob = heartModel.predict_proba(to_predict)

        x= str(prob[0][0])

        x= float(x)
        
    return jsonify(({"state":int(result[0]),"prob":x*100}))
        