from flask import Flask,render_template, request

import mysql.connector

from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import random
import scipy

import warnings
import joblib
import random
import os

app = Flask(__name__)

#database variables
fraud_db = mysql.connector.connect(
    database = "fraud_data",
    host = "localhost",
    port = "8889",
    user = "root",
    password = "root"
    )
fraud_cursor = fraud_db.cursor()
sql_insert = "INSERT INTO claims_data (ph_gender, car_model, replace_value, area_code, accident_type, days_to_expiry, fraud_prob, fraudulent) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
sql_select_train = "SELECT * from claims_data ORDER BY id DESC LIMIT 10000"
sql_select_view = "SELECT * from claims_data ORDER BY id DESC LIMIT 10"

#data preparation variables
car_model_list = ['A', 'B', 'C', 'D', 'E']
area_code_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
accident_list = ['D', 'F', 'T']

#num_of_models = len(car_model_list)
#num_of_codes = len(area_code_list)
#num_of_accidents = len(accident_list)

min_repl_value = 1000
max_repl_value = 5000

ohe = OneHotEncoder()
min_max_scaler = MinMaxScaler()

dirRawData = "./dataset/"
dirPModels = "./models/"

#required functions
def gender():
    gender = np.random.randint(2)
    return gender

def car_model(models):
    model_ = random.choice(models)
    return model_

def replace(low,high):
    #rep = round(np.random.uniform(low,high),0)
    rep = np.random.randint(low, high+1)
    #if rep >= (((high - low)*0.9) + low):
    #    rep_value = 1
    #else:
    #    rep_value = 0
    return rep

def area(areas):
    code = random.choice(areas)
    return code

def accident(accidents):
    accident = random.choice(accidents)
    return accident

def days():
    days = np.random.randint(366)
    #if days <= 100:
    #    days_value = 0
    #else:
    #    days_value = 1
    return days

def data_generator():
    #generate claim
    gender_value = gender()  
    model_value = car_model(car_model_list) 
    replace_value = replace(min_repl_value, max_repl_value)
    area_value = area(area_code_list)
    accident_value = accident(accident_list)
    day_value = days()
    
    #prepare the dataframe
    data = [[gender_value, model_value, replace_value, area_value, accident_value, day_value]]
    data = pd.DataFrame(data, columns=['ph_gender','car_model','replace_value','area_code', 'accident_type','days_to_expiry'])
    
    return data

def prep_train_dataset(dataframe):
    #encode categorical features
    cat_data = dataframe.filter(['car_model','area_code','accident_type'], axis=1).values.astype(str)
    ohe_fit = ohe.fit(cat_data)
    ohe_data = ohe_fit.transform(cat_data)
    ohe_data = scipy.sparse.csr_matrix.todense(ohe_data)
    ohe_df = pd.DataFrame(ohe_data, columns = ('model_A','model_B','model_C','model_D','model_E','area_A','area_B','area_C','area_D','area_E','area_F','area_G','area_H','area_I','area_J','acc_D','acc_F','acc_T'))
    joblib.dump(ohe_fit, dirPModels + 'ohe_fit.pkl')

    #standardize numerical data
    day_values = dataframe['days_to_expiry'].values.reshape(-1, 1)
    day_fit = min_max_scaler.fit(day_values)
    joblib.dump(day_fit, dirPModels + 'day_fit.pkl')
    day_transformed = day_fit.transform(day_values)
    day_df = pd.DataFrame(day_transformed, columns = ['days_to_expiry'])

    rep_values = dataframe['replace_value'].values.reshape(-1, 1)
    rep_fit = min_max_scaler.fit(rep_values)
    joblib.dump(rep_fit, dirPModels + 'rep_fit.pkl')
    rep_transformed = rep_fit.transform(rep_values)
    rep_df = pd.DataFrame(rep_transformed, columns = ['replace_value'])
    
    #combine the data
    gender_df = dataframe.filter(['ph_gender'], axis=1)
    #day_df = dataframe.filter(['days_to_expiry'], axis=1)
    #rep_df = dataframe.filter(['replace_value'], axis=1)
    fraud_df = dataframe.filter(['fraudulent'], axis=1)
    
    frames = [gender_df,rep_df,day_df,ohe_df,fraud_df]
    dataset = pd.concat(frames, axis=1)
    input_features = dataset.columns.values[0:-1]

    #split the data
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,-1].values
    y = y.reshape((len(y), 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2020)
    
    return X_train, X_val, y_train, y_val

def prep_claim_pred(claim, ohe, day_scale, rep_scale):
    #encode the categorical features
    cat_data = claim.filter(['car_model','area_code','accident_type'], axis=1).values.astype(str)
    ohe_data = ohe.transform(cat_data)
    ohe_data = scipy.sparse.csr_matrix.todense(ohe_data)
    ohe_df = pd.DataFrame(ohe_data, columns = ('model_A','model_B','model_C','model_D','model_E','area_A','area_B','area_C','area_D','area_E','area_F','area_G','area_H','area_I','area_J','acc_D','acc_F','acc_T'))

    #standardize numerical data
    day_value = claim['days_to_expiry'].values.reshape(-1, 1)
    day_transformed = day_scale.transform(day_value)
    day_df = pd.DataFrame(day_transformed, columns = ['days_to_expiry'])

    rep_value = claim['replace_value'].values.reshape(-1, 1)
    rep_transformed = rep_scale.transform(rep_value)
    rep_df = pd.DataFrame(rep_transformed, columns = ['replace_value'])

    #pick the other features
    gender_df = claim.filter(['ph_gender'], axis=1)
    #day_df = claim.filter(['days_to_expiry'], axis=1)
    #rep_df = claim.filter(['replace_value'], axis=1)

    frames = [gender_df,rep_df,day_df,ohe_df]
    dataset = pd.concat(frames, axis=1)

    return dataset

@app.route("/")
@app.route("/index")
def home():
    return render_template('home.html') 

@app.route("/train", methods=['POST', 'GET'])
def train():

    #retrive training data from sql db
    data = pd.read_sql(sql_select_train, fraud_db)

    #save the data as a csv
    data.to_csv(dirRawData + 'data.csv')

    #prepare the data for training
    X_train, X_val, y_train, y_val = prep_train_dataset(data)

    #train the model using SVC
    svc_model = SVC(C=1.0, 
             kernel='linear',
             class_weight='balanced', 
             probability=True,
             random_state=2020)
    svc_model.fit(X_train, y_train)
    
    #save model
    joblib.dump(svc_model, dirPModels + 'svc_model.pkl')
    #joblib.dump(ohe_fit, dirPModels + 'ohe_fit.pkl')

    return render_template('home.html')

@app.route('/claim', methods=['POST', 'GET'])
def claim():
    #import saved models
    svc_model = joblib.load('./models/svc_model.pkl')
    ohe_fit = joblib.load('./models/ohe_fit.pkl')
    day_fit = joblib.load('./models/day_fit.pkl')
    rep_fit = joblib.load('./models/rep_fit.pkl')

    #generate a new claim
    claim = data_generator()

    #prepare claim for prediction
    dataset = prep_claim_pred(claim, ohe_fit, day_fit, rep_fit)

    #run prediction model
    pred_value = int(svc_model.predict(dataset)[0])
    pred_prob = float(np.round((svc_model.predict_proba(dataset)[0][1])*100,0))

    #prepare variables to be saved to the db
    p_g = int(claim['ph_gender'].iloc[0])
    c_m = claim['car_model'].iloc[0]
    r_v = int(claim['replace_value'].iloc[0])
    a_c = claim['area_code'].iloc[0]
    a_t = claim['accident_type'].iloc[0]
    d_e = int(claim['days_to_expiry'].iloc[0])
    
    #save the claim to sql db
    val = (p_g, c_m, r_v, a_c, a_t, d_e, pred_prob, pred_value)
    fraud_cursor.execute(sql_insert, val)
    fraud_db.commit()
    
    #fetch data from mysql
    fraud_cursor.execute(sql_select_view)
    data = fraud_cursor.fetchall()

    return render_template('home.html', value = data)

if __name__ == '__main__':
    app.run()