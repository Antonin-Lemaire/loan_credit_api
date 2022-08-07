import json
from flask import jsonify
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
import pickle
from common_functions import Instance, ClassifierAPI

import lime
import dill
from pydantic import BaseModel

app = FastAPI()

with open('api_model.pkl', 'rb') as pickle_in:
    clf = dill.load(pickle_in)
print('loading model complete')
df = pd.read_csv('test.csv')
feats = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1',
         'EXT_SOURCE_2', 'EXT_SOURCE_3', 'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Higher education',
         'BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_UPDATE_MEAN',
         'BURO_CREDIT_ACTIVE_Active_MEAN', 'BURO_CREDIT_ACTIVE_Closed_MEAN',
         'PREV_NAME_CONTRACT_STATUS_Approved_MEAN', 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
         'PREV_CODE_REJECT_REASON_XAP_MEAN', 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
         'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'CC_CNT_DRAWINGS_CURRENT_MAX']
df = df[feats]
print('loading dataset complete')


class ClientInput(BaseModel):
    id: int


class Prediction(BaseModel):
    pred: float


@app.get('/')
def index():
    return {'message': 'Loan default prediction model'}


@app.get("/receive_index")
async def receive_index():
    return jsonify({'list_of_ids': df.index})


@app.post('/predict')
async def predict_outcome(data: ClientInput):
    print(data)
    unique_id = data.id

    df_input = df.iloc[unique_id, :]
    df_input.columns = feats
    print(df_input)

    prediction = clf.predict([df_input])
    context = clf.explain(df_input)
    answer = {"prediction": prediction,
              "context": context}

    return jsonify(answer)


@app.get('/get_neighbours')
async def neighbours(client_id):
    client_input = df.iloc[client_id, :]
    client_input.columns = df.columns




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
