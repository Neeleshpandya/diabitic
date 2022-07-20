# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:11:34 2022

@author: NH1305
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


# 2. Class which describes diabetes prediction

class diabetes(BaseModel):
    pregnancies: int 
    glucose: int 
    bloodPressure: int 
    SkinThickness: int
    Insulin: int
    BMI:float
    DiabetesPedigreeFunction:float
    Age:int
    
@app.post('/predict')
def predict_diabetes(data:diabetes):
    data = data.dict()
    pregnancies=data['pregnancies']
    glucose=data['glucose']
    bloodPressure=data['bloodPressure']
    SkinThickness=data['SkinThickness']
    Insulin=data['Insulin']
    BMI=data['BMI']
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    Age=data['Age']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[pregnancies,glucose,bloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    output = prediction[0]
    if output == 0:
        prediction="Diabetes : No"
    else:
        prediction="Diabetes : yes"
    return {
        'prediction': prediction
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
