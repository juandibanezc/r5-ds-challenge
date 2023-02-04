from predict import run_prediction_pipe
from train import run_training_pipe
from flask import request
from app import app
import os


@app.route("/training", methods=['GET'])
def pycaret_train():
  
    if request.method == 'GET':
      return run_training_pipe()
    else:
      return 'Wrong method, only GET requests are accepted for this endpoint.', 400
    
@app.route("/prediction", methods=['POST'])
def pycaret_pred():
  
    if request.method == 'POST':
      return run_prediction_pipe(request)
    else:
      return 'Wrong method, only POST requests are accepted for this endpoint.', 400

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8080)))
