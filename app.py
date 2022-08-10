#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:45:31 2022

@author: abhinav
"""

from flask import Flask, render_template, request
import numpy as np 
import pickle

from sklearn.preprocessing import StandardScaler



app = Flask(__name__, template_folder='template')


model = pickle.load(open("model.pkl","rb"))

scaler = pickle.load(open("scaler.pkl","rb"))

@app.route("/",methods=["GET"])
def home():
    return render_template("/index.html")


@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        
        km_driven = request.form['km_driven']
        
        owner = request.form['owner']
        if owner == "First":
            owner = 0
        elif owner == "Second":
            owner = 1
        elif owner == "Third":
            owner = 3
        elif owner == "Fourth":
            owner = 2
        else:
            owner = 4
        
        car_age = int(request.form['year'])
        car_age = 2022 - car_age
        

     
        fuel_type = request.form['fuel_type']
        if fuel_type == "Diesel":
            fuel_Diesel = 1	
            fuel_LPG = 0
            fuel_Petrol = 0
        elif fuel_type == "Petrol":
            fuel_Diesel = 0	
            fuel_LPG = 0
            fuel_Petrol = 1 
        elif fuel_type == "CNG":
            fuel_Diesel = 0	
            fuel_LPG = 0
            fuel_Petrol = 0
        elif fuel_type == "LPG":
            fuel_Diesel = 0	
            fuel_LPG = 1
            fuel_Petrol = 0 
        
        seller_type = request.form['seller_type']
        if seller_type == "Individual":
            seller_type_Individual	= 1
            seller_type_Trustmark_Dealer = 0
        elif seller_type =="Dealer":
            seller_type_Individual	= 0
            seller_type_Trustmark_Dealer = 0
        else:
            seller_type_Individual	= 0
            seller_type_Trustmark_Dealer = 1
            
        
        transmission = request.form['transmission']
        if transmission == "Manual":
            transmission_Manual = 1 
        else:
            transmission_Manual = 0
        
        features = np.array([km_driven, owner, car_age, fuel_Diesel, fuel_LPG,fuel_Petrol, seller_type_Individual, seller_type_Trustmark_Dealer, transmission_Manual]).reshape(1,-1)
        
        
        prediction = model.predict(features)
        
        prediction = scaler.inverse_transform(prediction.reshape(1,-1))
        
        return render_template("/index.html", prediction_text="THE SELLING PRICE IS AROUND "+str(prediction[0][0]))

if __name__ == '__main__':
    app.run(debug=True)
        
        
