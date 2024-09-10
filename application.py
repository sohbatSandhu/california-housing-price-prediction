from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import HousingData, PredictionPipeline

application = Flask(__name__)

app = application

# route for a home page

@app.route('/california-housing')
def index():
    return render_template('home.html')

@app.route('/california-housing/predict', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction-form.html')
    else:
        data= HousingData(
            longitude = float(request.form.get('longitude')),
            latitude  = float(request.form.get('latitude')),
            housing_median_age = float(request.form.get('housing_median_age')),
            total_rooms  = float(request.form.get('total_rooms')),
            total_bedrooms  = float(request.form.get('total_bedrooms')),
            population  = float(request.form.get('population')),
            households  = float(request.form.get('households')),
            median_income  = float(request.form.get('median_income')),
            ocean_proximity = request.form.get('ocean_proximity')
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        print(data.households)
        return render_template('result.html', results = results[0], data = data)
    
if __name__ == "__main__":
    app.run(debug=True)