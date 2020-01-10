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

Three files obtained from the model training excercise will be available on the server : Check link https://github.com/femtonelson/Classifying-Songs-Genres-From-Audio-Data
- data_preprocess.py : Contains a function which pre-processes an input pandas dataframe and returns a numpy array with the reduced features
                       Columns expected in the input dataframe : track_id, acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence
- logreg.pkl : The trained Logistic Regression model
- logreg_columns.pkl : The column names of the input dataframe

# Model Deployment Procedure

- Setup AWS EC2 instance in a public subnet with a public IP address
- Install and configure Flask on this machine, to be accessible on suitable port
- Run Postman API client, send requests to the server to obtain predictions


# Setup a publicly accessible AWS EC2 Instance - Flask API Server




# Flask API Configuration




# Testing the API in API client - Postman


































Useful links : 
- https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
- https://blog.cloudera.com/putting-machine-learning-models-into-production/
