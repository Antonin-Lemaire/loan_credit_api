import json
import numpy as np
import dill
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from io import BytesIO
from starlette.responses import StreamingResponse

app = FastAPI()

with open('api_model.pkl', 'rb') as pickle_in:
    clf = dill.load(pickle_in)
print('loading model complete')
feats = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT',
         'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
         'EXT_SOURCE_3', 'NAME_INCOME_TYPE_Working',
         'NAME_EDUCATION_TYPE_Higher education', 'BURO_DAYS_CREDIT_MIN',
         'BURO_DAYS_CREDIT_MEAN', 'BURO_DAYS_CREDIT_UPDATE_MEAN',
         'BURO_CREDIT_ACTIVE_Active_MEAN', 'BURO_CREDIT_ACTIVE_Closed_MEAN',
         'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',
         'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
         'PREV_CODE_REJECT_REASON_XAP_MEAN', 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
         'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', 'CC_CNT_DRAWINGS_CURRENT_MAX']


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ClientInput(BaseModel):
    DAYS_BIRTH: int
    DAYS_EMPLOYED: float
    REGION_RATING_CLIENT: int
    REGION_RATING_CLIENT_W_CITY: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    NAME_INCOME_TYPE_Working: int
    NAME_EDUCATION_TYPE_Higher_education: int = Field(
        ..., alias='NAME_EDUCATION_TYPE_Higher education'
    )
    BURO_DAYS_CREDIT_MIN: float
    BURO_DAYS_CREDIT_MEAN: float
    BURO_DAYS_CREDIT_UPDATE_MEAN: float
    BURO_CREDIT_ACTIVE_Active_MEAN: float
    BURO_CREDIT_ACTIVE_Closed_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Approved_MEAN: float
    PREV_NAME_CONTRACT_STATUS_Refused_MEAN: float
    PREV_CODE_REJECT_REASON_XAP_MEAN: float
    PREV_NAME_PRODUCT_TYPE_walk_in_MEAN: float = Field(
        ..., alias='PREV_NAME_PRODUCT_TYPE_walk-in_MEAN'
    )
    CC_CNT_DRAWINGS_ATM_CURRENT_MEAN: float
    CC_CNT_DRAWINGS_CURRENT_MAX: float


@app.get('/')
def index():
    return {'message': 'Loan default prediction model'}


@app.post('/predict')
async def predict_outcome(data: ClientInput):
    data_dict = jsonable_encoder(data)
    for key, value in data_dict.items():
        data_dict[key] = [value]
    df_input = pd.DataFrame.from_dict(data_dict)
    prediction = clf.predict(df_input)

    return json.dumps(prediction.tolist())


@app.post('/context_map')
async def give_context(data: ClientInput):
    data_dict = jsonable_encoder(data)
    for key, value in data_dict.items():
        data_dict[key] = [value]
    df_input = pd.DataFrame.from_dict(data_dict)
    context = clf.explain(df_input.T)
    map_context = dict(context.as_map())
    return json.dumps(map_context, cls=NpEncoder)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
