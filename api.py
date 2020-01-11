# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
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

            # Convert the output pack to JSON and send to client
            return output_df.to_json(orient='records')

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

def data_preprocess(input_df):
    # This function processes an input pandas dataframe and returns a numpy array with the reduced features
    # Columns expected : track_id, acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence

    # Filter out quantitative features and drop 'track_id'
    features = input_df.drop(columns=['track_id'])

    # Import the StandardScaler class and instanciate it
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Scale the quantitative features i.e replace each value of a feature x by [(value-mean(x))/std(x)]
    scaled_train_features = scaler.fit_transform(features)

    # Import PCA class
    from sklearn.decomposition import PCA

    # Perform PCA with the chosen number of components = 6 and project data onto components
    n_components = 6
    pca = PCA(n_components, random_state=10)
    pca.fit(scaled_train_features)
    pca_projection = pca.transform(scaled_train_features)
    # Return the reduced numpy array with n_components = 6 features
    return pca_projection

if __name__ == '__main__':

    logreg = joblib.load("logreg.pkl") # logreg.pkl"
    print ('Model loaded')
    model_columns = joblib.load("logreg_columns.pkl") # Load "logreg_columns.pkl"
    print ('Model columns loaded')

    # Run FLask on port 5500 and listen to requests from all IP addresses
    app.run(port = 5500, debug=True, host= '0.0.0.0')