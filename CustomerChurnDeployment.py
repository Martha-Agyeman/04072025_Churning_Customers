import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Customer Churn Predictor')

model = "C:/Users/Martha Agyeman/Downloads/finalChurnModel_mlp.pkl"

md = pickle.load(open(model, 'rb'))


SeniorCitizen= st.number_input('SeniorCitizen: (Enter 0 or 1)')
tenure= st.number_input('tenure (Enter a number)')
MonthlyCharges= st.number_input('MonthlyCharges (Enter a number)')
gender= st.text_input('gender (Male or Female)')
Partner= st.text_input('Partner (Enter Yes or No)')
Dependents= st.text_input('Dependents (Enter Yes or No)')
PhoneService= st.text_input('PhoneService (Enter Yes or No)')
MultipleLines= st.text_input('Multiple Lines (Enter Yes, No or No phone service)')
InternetService= st.text_input('IntrnetService (Enter DSL Fiber optic or No)')
OnlineSecurity= st.text_input('OnlineSecurity (Enter Yes, No internet service or No)')
OnlineBackup= st.text_input('OnlineBackup (Enter Yes, No internet service or No)')
DeviceProtection= st.text_input('DeviceProtection (Enter Yes, No internet service or No)')
TechSupport= st.text_input('TechSupport (Enter Yes, No internet service or No)')
StreamingTV = st.text_input('StreamingTV (Enter Yes, No internet service or No) ')
StreamingMovies= st.text_input('StreamingMovies (Enter Yes, No internet service or No)')
Contract= st.text_input('Contract (Enter Month-to-month, Two year or One year)')
PaperlessBilling= st.text_input('PaperlessBilling (Enter Yes or No)')
PaymentMethod= st.text_input('PaymentMethod (Enter Electronic check, Mailed check, Bank transfer (automatic) or Credic card (automatic))')
TotalCharges= st.text_input('Total Charges (Enter a number)')

if st.button('Predict'):
    input_data = np.array([[SeniorCitizen, tenure, MonthlyCharges, ...]])
    prediction = md.predict(input_data)

    st.write("The predicted churn for this customer is ", prediction[0])
