from flask import Flask,request, render_template
from sagemaker.serializers import CSVSerializer
import numpy as np
import json
import sagemaker
#from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import Session

sess = sagemaker.Session()
endpoint_name='xgboost-2021-11-24-03-55-25-924' # Replace with your endpoint
predictor_model = sagemaker.predictor.Predictor(endpoint_name, sagemaker_session=sess)
predictor_model.serializer = CSVSerializer() 

app = Flask(__name__)
@app.route('/api', methods=['POST'])
def parse_request():
    data = request.form

    flat_type = int(data["flat_type"])
    floor_area_sqm = int(data["floor_area_sqm"])
    lease_commence_date = int(data["lease_commence_date"])
    storey_range = int(data["storey_range"])
    town = int(data["town"])

    test_array =  np.zeros((1, 30))
    test_array[0][0] = flat_type
    test_array[0][1] = floor_area_sqm
    test_array[0][2] = lease_commence_date
    test_array[0][3] = storey_range
    test_array[0][town+3] = 1 

    prod_prediction = predictor_model.predict(test_array).decode('utf-8') # predict!
    return json.dumps({"result": "OK", "prediction": prod_prediction})

@app.route("/")
def index():
   return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)