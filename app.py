import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__) 
model = joblib.load("breast_cancer_classification.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    if (output==1):
       return render_template('index.html', prediction_text='It seems like you have CANCEROUS CELLS (MALIGNANT)')

    else:
       return render_template('index.html', prediction_text='DONT WORRY ,you dont have Breast Cancer')

if __name__ == "__main__":
    app.run(debug=True,port=8000)