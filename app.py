#Building the webapp using FastAPI for the predictor model

# 1. importing dependencies
import uvicorn ##ASGI (Allows paraller computing)
from fastapi import FastAPI , Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates 
from BankNotes import BankNote #Calling the bank note measurement class
import numpy as np
import pandas as pd
import pickle 

# 2. Create app object and setting up the templates
app = FastAPI()
# templates = Jinja2Templates(directory="templates")

#Loading the saved model 
pickle_in = open('classifier.pkl','rb')
clf=pickle.load(pickle_in)

# 3. Index route (First page after url opens)
@app.get('/')
def index():
    return {'message':'Hello'}

# 4. Prediction functionality API . Expects a input json data 
# and returns the predicted bank note with confidence(probabilities)
@app.post('/predict')
# data holds the input data which is automatically mapped into the BankNote Class measurements 
def predict_banknote(data:BankNote):
    data = data.dict() #converting data into a dictionary
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    prediction = clf.predict([[variance,skewness,curtosis,entropy]])
    if (prediction[0]>0.5):
        prediction= "Its a Dupe"
    else:
        prediction = "Its Legit"  
    return {'prediction':prediction}     

# 5. Run API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1',port=8000) #local address