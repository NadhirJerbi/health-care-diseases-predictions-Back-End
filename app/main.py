from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS, cross_origin
import pickle
import os

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
    return result[0]


@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        print(request.form.to_dict())
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"

    return jsonify(str("Class  " + prediction))


