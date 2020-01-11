# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Import the data pre-processing toolbox function from "data_preprocess.py"
from data_preprocess import data_preprocess

# Create API instance
app = Flask(__name__)

@app.route('/predict', methods=['POST'])  # This decorator indicates which URL "/predict" should trigger the execution of the func predict
def predict():
    if logreg:
        try:
            # Read input json request into a dataframe and fill dummy values
            json_ = request.json

            # Print the content of JSON request to server terminal
            print(json_)

            input_df = pd.DataFrame(json_)

            # Process the input dataframe and return a reduced numpy array
            reduced_arr = data_preprocess(input_df)

            # Run the prediction on the reduced array
            prediction = list(logreg.predict(reduced_arr))

            # Create an output dataframe to contain track_id and prediction for each track id
            output_df = input_df[['track_id']]
            output_df['prediction'] = prediction

            # Print result in server terminal
            print(output_df)

            # Convert the output to JSON and send to client
            return output_df.to_json(orient='records')

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':  # To be executed when "sudo python3 api.py" is run in server terminal

    logreg = joblib.load("logreg.pkl") # logreg.pkl"
    print ('Model loaded')
    model_columns = joblib.load("logreg_columns.pkl") # Load "logreg_columns.pkl"
    print ('Model columns loaded')

    # Run FLask on port 5500 and listen to requests from all IP addresses
    app.run(port = 5500, debug=True, host= '0.0.0.0')