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

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        result = heartModel.predict(to_predict)
        prob = heartModel.predict_proba(to_predict)
        x= str(prob[0][0])
        x= int(x)
       
    return {'state':result[0],'prob':x*100 }


@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        
        to_predict_list = request.json
        
        to_predict_list = list(to_predict_list.values())
        print('therekkldkljdkjdf')
        print(to_predict_list)
        to_predict_list = list(map(float, to_predict_list))
        to_predict = np.array(to_predict_list).reshape(1,7)
        result = heartModel.predict(to_predict)
        prob = heartModel.predict_proba(to_predict)
        x= str(prob[0][0])
        x= float(x)
        print('(////////////////////////////////////////////)')
        print(type(x))
    return jsonify(({"state":int(result[0]),"prob":x*100}))
        

    


app.run(debug=True)