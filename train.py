from imblearn.combine import SMOTEENN
from pycaret.classification import *
from google.cloud import storage
from dotenv import load_dotenv
from datetime import datetime
from typing import Tuple
from app import app
import psycopg2
import pickle
import time
import os

load_dotenv()
t0 = time.time()
today = datetime.now().strftime('%Y%m%dT%H%M%S')

def run_training_pipe() -> Tuple[str, int]:
    app.logger.info("Starting training")

    try:
      app.logger.info("Running query to PostgreSQL database")
      conn = psycopg2.connect(database="juan-database",
                        host="34.132.75.146",
                        user="postgres",
                        password=os.environ.get('PASSWORD'),
                        port="5432")

      query = """ SELECT 
                    policynumber AS id,
                    accidentarea AS accident_area,
                    maritalstatus AS marital_status,
                    policytype AS policy_type,
                    agenttype AS agent_type,
                    numberofsuppliments AS number_of_suppliments,
                    numberofcars AS number_of_cars,
                    vehicleprice AS vehicle_price,
                    ageofvehicle AS age_of_vehicle,
                    ageofpolicyholder AS age_of_policy_holder,
                    fault AS fault,
                    make AS make,
                    deductible AS deductible,
                    fraudfound_p AS fraud
                  FROM 
                    public.fraudes
              """
      df = pd.read_sql(query, conn)

      app.logger.info("Getting unseen data for validation")
      data = df.sample(frac=0.95, random_state=785)
      data_unseen = df.drop(data.index)
      data.reset_index(inplace=True, drop=True)
      data_unseen.reset_index(inplace=True, drop=True)

      app.logger.info('Data for modeling: ' + str(data.shape))
      app.logger.info('Unseen data for predictions: ' + str(data_unseen.shape))

      app.logger.info("Setting up experiment")
      sme = SMOTEENN(random_state=42, n_jobs=-1)
      s = setup(data = data, 
                session_id=123,
                remove_outliers = True, 
                outliers_threshold = 0.05, 
                normalize = True, 
                transformation = True, 
                combine_rare_levels = True, 
                ignore_low_variance = True, 
                remove_multicollinearity = True, 
                multicollinearity_threshold = 0.7, 
                data_split_stratify = True, 
                fix_imbalance=True, 
                fix_imbalance_method = sme, 
                high_cardinality_features = ['make'], 
                high_cardinality_method = 'frequency', 
                target = 'fraud', 
                ignore_features = ['id'], 
                categorical_features = ['accident_area','marital_status','fault',
                                        'policy_type','agent_type','number_of_suppliments',
                                        'number_of_cars','vehicle_price','age_of_vehicle',
                                        'age_of_policy_holder'],
                numeric_features = ['deductible'], 
                silent = True)

      app.logger.info("Getting best model trained")
      best_model = compare_models(cross_validation=False, n_select = 1, include = ['lr', 'lda','svm','ridge'])

      app.logger.info("Tunning hyperparameters for best model")
      tuned_best_model = tune_model(best_model, n_iter = 10, fold = 10, optimize = 'AUC', search_library = "scikit-optimize", search_algorithm = "bayesian", choose_better = True)

      app.logger.info("Displaying average metrics of the cross validation tunnig of the best model")
      model_metrics = pull()
      metrics = model_metrics.loc['Mean']
      print(model_metrics.loc['Mean'])

      app.logger.info("Predicting unseen data for validation")
      predictions = predict_model(tuned_best_model, data=data_unseen, raw_score=True)

      app.logger.info("Displaying metrics for the validation data")
      validation_metrics = pull()
      print(validation_metrics)

      app.logger.info("Training model on the whole dataset")
      final_model = finalize_model(tuned_best_model)
     
      app.logger.info('Saving model')
      storage_client = storage.Client()
      bucket = storage_client.bucket('grupor5-fraud-detection')
      blob = bucket.blob(f'models/model_{today}.pkl')
      blob.upload_from_string(data=pickle.dumps(final_model))

      app.logger.info('Saving setup configuration')
      save_config('./models/model_setup.pkl')
      blob = bucket.blob(f'model_setup/model_setup_{today}.pkl')
      blob.upload_from_filename('./models/model_setup.pkl')

      t1 = time.time()
      app.logger.info(f'Training completed after {round(t1 - t0)} seconds')
      return "Training completed!" , 200
      
    except Exception as e:
      app.logger.error(f'Catched an exception during training of the model: {e}')
      return "Training failed!" , 500
    
