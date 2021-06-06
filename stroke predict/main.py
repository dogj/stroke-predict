from flask import Flask, render_template, request
import numpy as np
from joblib import dump, load
import joblib


model = joblib.load('test_model_balanced_random_forest-29.sav')

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['gender']
    data2 = request.form['age']
    data3 = request.form['hypertension']
    data4 = request.form['heart_disease']
    data5 = request.form['ever_married']
    data6 = request.form['work_type']
    data7 = request.form['Residence_type']
    data8 = request.form['bmi']
    data9 = request.form['smoking_status']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9]])
    pred = model.predict(arr)
    prob = model.predict_proba(arr)
    return render_template('after.html', data=pred,data2=prob)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)















