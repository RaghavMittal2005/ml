from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sys
from src.pipeline.predict_pipeline import PredictData,CustomData
application=Flask(__name__)
app=application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        dt=data.get_data()
        pred=PredictData()
        results=pred.pred(dt)
        print("after Prediction")
        return render_template('home.html',results=results[0])

        


if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
    