from flask import Flask, request
import pandas as pd 
import numpy as np
import pickle 
import flasgger
from flasgger import Swagger


#Initializing the app instance
app=Flask(__name__)
Swagger(app)
#Loading the classifier
pickle_in=open('classifier.pkl','rb')
clf=pickle.load(pickle_in)

#Home page api
@app.route('/')
def index():
    return "Welcome all"

#predict page api
@app.route('/predict',methods=["Get"])
def predict_note():
    """Let's find out if your bank note is legit or fake.
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true  
    responses:
        200:
            description: The output values     
    """
    variance=request.args.get('variance')    
    skewness=request.args.get('skewness')    
    curtosis=request.args.get('curtosis')    
    entropy=request.args.get('entropy')
    prediction=clf.predict([[variance,skewness,curtosis,entropy]])  
    return "The predicted value is: " + str(prediction)  


#Providing values that need to be predicted
@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's find out if your bank note is legit or fake.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: The output values     
    """
    df_test=pd.read_csv(request.files.get('file'))
    prediction=clf.predict(df_test)
    return "The predicted values are" + str(list(prediction))


if __name__ == "__main__":
    app.run()