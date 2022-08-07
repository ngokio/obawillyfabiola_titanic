import pandas as pd
from house_function import first_letter
import joblib
from flask import Flask,request
pipeline=joblib.load("titan.model")
import flask
print(flask.__version__)

#demarage de mon application
app=Flask("__name__")

#page d'accueil
@app.route('/')
def index():
  return "<h1>voici ma page de prediction pour le titanic</h1>"

@app.route('/ping',methods=['GET'])
def ping():
  return("pong",200)

@app.route('/predict',methods=['POST'])
def predict():
  df=pd.DataFrame(request.json)  
  resultat=pipeline.predict(df)[0]
  return (str(resultat),200)

#obligatoire pour demarer ma page
if __name__ == "__main__":
    app.run(host="0.0.0.0")


