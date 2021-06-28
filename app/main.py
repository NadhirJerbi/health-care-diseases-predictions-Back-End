from flask import Flask, jsonify
from flask import request
from numpy import np
import pickle
#from flask_cors import CORS, cross_origin

app = Flask(__name__)

#cros =CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'

model = pickle.load(open('/pkl/heart_model.pkl', 'rb'))

@app.route("/" )
#@cross_origin()
def home_view():
        return "<h1>Welcome si slim</h1>"

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = pickle.load(open('/pkl/heart_model.pkl', 'rb'))
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
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

