import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
import streamlit as st
import pickle

# This file will be used to make Streamlit web app

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the scalers and encoders
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

with open('label_encoders_gender.pkl','rb') as file:
    label_encoder = pickle.load(file)
    
with open('onehot_encoders_geo.pkl','rb') as file:
    onehot_encoder = pickle.load(file)
    

# Streamlit web app
st.title('Customer Churn Prediction')

# Input form

geography = st.selectbox('Geography', onehot_encoder.categories_[0]) # onehot_encoder['Geography'].categories_[0] is the list of unique values of Geography
gender = st.selectbox('Gender', label_encoder.classes_) # label_encoder.classes_ is the list of unique values of gender 
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.number_input('Tenure',0,10)
num_of_products = st.number_input('Number of Products',1,4)
has_credit_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Preprocess the input
input_data = pd.DataFrame({ # Create a DataFrame with the input data by putting them in a dictionary and then converting it to a DataFrame
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]], # here we are passing the encoded value of gender to the DataFrame [0] mea
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode the Geography in different dataframes 
geo = onehot_encoder.transform([[geography]]).toarray() # onehot_encoder.transform([[geography]]) will return a sparse matrix, so we convert it to an array
geo = pd.DataFrame(geo, columns=onehot_encoder.get_feature_names_out(['Geography'])) # Convert the array to a DataFrame and set the column names

# Concatenate the input data and the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True),geo],axis=1) # reset_index(drop=True) will reset the index of the input_data DataFrame and drop the old index

# Scale the input data
input_data = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data)
prediction_proba = prediction[0][0] # prediction is a 2D array, so we get the first element of the first element which is the probability of the customer churning


# Display the prediction
st.write('The probability of the customer churning is:',prediction_proba)

if prediction_proba > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')
