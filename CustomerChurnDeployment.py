import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title('Customer Churn Predictor')

md = load_model("Model.h5")

# Function to preprocess input data
def preprocess_input(data, label_encoder_dict=None, scaler=None):
    if label_encoder_dict is None:
        label_encoder_dict = {}
        scaler = StandardScaler()

    # Convert data to DataFrame
    columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender', 'Partner',
               'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod', 'TotalCharges']
    
    data_df = pd.DataFrame(data, columns=columns)

    # Encode categorical columns
    for column in data_df.select_dtypes(include=['object']).columns:
        if column not in label_encoder_dict:
            label_encoder_dict[column] = LabelEncoder()
        data_df[column] = label_encoder_dict[column].fit_transform(data_df[column])

    # Scale numerical features
    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data_df[numerical_columns] = scaler.fit_transform(data_df[numerical_columns])

    return data_df, label_encoder_dict, scaler

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
    input_data = np.array([[SeniorCitizen, tenure, MonthlyCharges, gender, Partner,
                            Dependents, PhoneService, MultipleLines, InternetService,
                            OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                            StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                            PaymentMethod, TotalCharges]])

    # Preprocess input data
    input_df, label_encoder_dict, scaler = preprocess_input(input_data)

    # Make prediction
    prediction = md.predict(input_df)

       # Print prediction
    churn_prediction = "Yes" if prediction[0] > 0.5 else "No"
    st.write("The predicted churn for this customer is ", churn_prediction)
