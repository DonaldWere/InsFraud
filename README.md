# InsFraud
This repository provides a machine learning code for checking likelihood of fraud in insurance claims. 

## Introduction
This project uses the SVC model to predict the likelihood that any reported claim is fraudulent. 

## Data 
10,000 claims with six features, were simulated and determined to be either fraudulent or not using a predetermined algorithm. 80% of the data was used for training on 3 different models, with the rest of the data being used to validated the prediction results. The models are (Neural Network, SVC and Logistic Regression). The SVC model perfomred best and was selected for the final predictions. 

## App
Flask has been used to build the model, with an SQL server and a web user interface. 
