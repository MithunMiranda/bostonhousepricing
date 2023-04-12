import pickle
from flask import Flask, request, app, jsonify,url_for, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app=Flask(__name__)
## load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    ## the data values receieved must be transformed into a list which is then changed into np array
    ## the numpy array is then reshaped to a single row, multiple columns(1,13)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    # since this is a two dimensional array, we need output[0]
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)
