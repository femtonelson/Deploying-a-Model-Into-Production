# Deploying-a-Model-Into-Production
This repository describes the steps taken to deploy a model trained and tested with scikit-learn python library. This model predicts if a given song is of type "Rock" or "Hip-Hop" based on certain features. Check link : https://github.com/femtonelson/Classifying-Songs-Genres-From-Audio-Data .This is done with a web service development framework in Python known as Flask.

# Introduction to Model Deployment
A model contains information on the type of algorithm used and relevant coefficients calculated during its training. One of the key requirements of a model is that 
it needs to be portable, so as to transfer functionality flawlessly from one environment to a different one. 

The majority of models are built in Python/R but the applications that consume these models may belong to different technology stacks. It is thus necessary to have an interface
between different technologies with a standard exhange format. APIs (Application Programming Interface) provide this linkage. They can run as web applications using Hypertext Transfer Protocol (HTTP) request messages, 
along with a definition of the structure of response messages, usually in an Extensible Markup Language (XML) or JavaScript Object Notation (JSON) format. 
In this excercise, the Flask API will be used. It has an inbuilt light-weight web server which needs minimal configuration, and it can be controlled from Python code.

Flask web server will be installed on a publicly accessible AWS instance and will be configured to respond to JSON requests (sent by a Postman API client) with song type predictions.  

# Model Deployment Procedure

- Setup AWS EC2 instance in a public subnet with a public IP address
- Install and configure Flask on this machine, to be accessible on port 5500 for example
- Run Postman API client and send requests to the server to obtain predictions

Three files obtained from the model training excercise should be available in the working directory on the server :
- data_preprocess.py : Provides the function which pre-processes an input pandas dataframe and returns a numpy array with the reduced features
- logreg.pkl : The trained Logistic Regression model
- logreg_columns.pkl : The column names of the input dataframe


# Setup a publicly accessible AWS EC2 Instance and install Flask API Server
```
# Install Flask and necessary Python packages
$sudo apt-get update
$sudo apt install python3-pip
$sudo pip3 install flask
$sudo pip3 install pandas
$sudo pip3 install joblib
$sudo pip3 install -U scikit-learn 
```

# Configure Flask application, configuration file : [api.py](/api.py)

```
$sudo python3 api.py
(base) ubuntu@ip-10-0-4-193:~$ sudo python3 api.py
Model loaded
Model columns loaded
 * Serving Flask app "api" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:5500/ (Press CTRL+C to quit)
 * Restarting with stat
Model loaded
Model columns loaded
 * Debugger is active!
 * Debugger PIN: 150-894-586
```


# Send JSON Requests to Flask in API client - Postman

- Download Postman : https://www.getpostman.com/downloads/

- Set 'POST' method for forwarding JSON requests as specified in "api.py" and provide the URL to the prediction server : http://PublicIP.Of.EC2.Instance:5500/predict

- Set the Request Format as "JSON".

NB : For principal component analysis algorithm to work, the number of components chosen in the training process : n_components = 6 must not exceed min(n_samples, n_features).
Given the number of features is 8 : acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence
We must provide at least 06 samples in a prediction request on Postman.

- [Example of Request](/Example-of-request.json), [Example of Response](/Example-of-response.json)
<img src="./postman-request-response.jpg">




































Useful links : 
- https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
- https://blog.cloudera.com/putting-machine-learning-models-into-production/
