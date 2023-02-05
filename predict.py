from pycaret.classification import *
from typing import Tuple
from app import app
import pickle
import gcsfs
import time


t0 = time.time()

def run_prediction_pipe(request) -> Tuple[dict, int]:
    app.logger.info("Starting prediction")

    try:
      request_data = request.get_json()
      df = pd.DataFrame.from_dict(request_data)

      app.logger.info("Loading setup configuration")
      # Cloud
      fs = gcsfs.GCSFileSystem()
      file = fs.ls('grupor5-fraud-detection/model_setup')[-1]
      with fs.open(file, "rb") as f:
        s = load_config(f)
      
      # Local 
      # with open("./models/model_setup.pkl", "rb") as f:
      #   s = load_config(f)
      
      app.logger.info("Loading trained model")
      # Cloud
      file = fs.ls('grupor5-fraud-detection/models')[-1]
      with fs.open(file, "rb") as f:
        model = pickle.load(f)

      # Local
      # with open("./models/model.pkl", "rb") as f:
      #   model = pickle.load(f)
    
      app.logger.info("Making predictions")
      predictions = predict_model(model, data = df.drop(['fraud'],axis=1), raw_score=True)

      app.logger.info("Transforming prediction results to json")
      df_result = predictions[['id','Label','Score_0','Score_1']]
      print(f"Predictions:\n")
      print(df_result)
      json_result = df_result.to_json(orient='records', lines=True)

      t1 = time.time()
      app.logger.info(f'Prediction completed after {round(t1 - t0)} seconds')
      return json_result , 200
      
    except Exception as e:
      app.logger.error(f'Catched an exception during prediction pipeline: {e}')
      return "Prediction failed!" , 500