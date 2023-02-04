from pycaret.classification import *
from typing import Tuple
from app import app
import pickle
import json
import time

t0 = time.time()

def run_prediction_pipe(request) -> Tuple[dict, int]:
    app.logger.info("Starting prediction")

    try:
      request_data = request.get_json()
      df = pd.DataFrame.from_dict(request_data)

      app.logger.info("Loading pipeline setup")
      with open("./models/model_setup.pkl", "rb") as f:
        s = load_config(f)
      
      app.logger.info("Loading trained model")
      with open("./models/model.pkl", "rb") as f:
        model = pickle.load(f)
    
      app.logger.info("Making predictions")
      predictions = predict_model(model, data = df.drop(['fraud'],axis=1), raw_score=True)
      print(f"The model prediction is: {predictions['Label'].iloc[0]}")

      app.logger.info("Transforming prediction results to json")
      df_result = predictions[['Label','Score_0','Score_1']]
      df_result.index = ['prediction']
      json_result = df_result.to_json(orient="index")

      t1 = time.time()
      app.logger.info(f'Prediction completed after {round(t1 - t0)} seconds')
      return json_result , 200
      
    except Exception as e:
      app.logger.info(f'Catched an exception during prediction pipeline: {e}')
      return "Prediction failed!" , 500
    
