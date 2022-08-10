# Loan_credit_api

Api hosted at heroku, running in tandem with the dashboard in streamlit

https://loan-credit-default-risk-api.herokuapp.com/

## Organisation

The main file is loan_api.py.

It will use functions from common_functions.py and use the api_model.pkl to do predictions and explanations.

Procfile is used to launch the server in Heroku.

## How to use locally

You will need requirements.txt, loan_api.py, common_functions.py and api_model.pkl.

First install all required libraries in python console with :

pip install -r requirements.txt

Then run in system console "loan_api.py" with :

python3 loan_api.py

The default local address is 127.0.0.1/8000/, though you can modify it in the source file by changing the parameters in

uvicorn.run(app, host='127.0.0.1', port=8000)


## General usage requests

/predict takes a 1-row pandas DataFrame with named columns of size (1, 20).

It will return a size (1, 2) array with the probability of either TARGET class.

0 means the client will not default, 1 means the client would.


/context_map takes the same 1-row pandas DataFrame as input as /predict.

It returns a dict containing a list of tuples. 

In each tuple with the schema [x, y]:

x is the index of your input DataFrame.columns.

y is the weight of this feature in the predict method.

