# -*- coding: utf-8 -*-
"""
@author: dekrk
"""

import streamlit as st
import numpy as np
import pickle

# Load the trained model
filename = 'random_forest_3.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to predict the outcome
def prediction(input_data):
    # Reshape the input data
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'Customer will not subscribe to term deposit'
    else:
        return 'Customer will subscribe to term deposit'

# Streamlit app
def main():
    # Title
    st.title('Term Deposit Subscription Prediction')
    
    # Get input data from user
    age = st.slider('Age', min_value=18, max_value=100, step=1)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
    education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'])
    default = st.selectbox('Credit in Default?', ['no', 'yes'])
    housing = st.selectbox('Housing Loan?', ['no', 'yes'])
    loan = st.selectbox('Personal Loan?', ['no', 'yes'])
    contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    day_of_week = st.selectbox('Select Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    duration = st.number_input('Last Contact Duration (seconds)')
    campaign = st.number_input('Number of Contacts During this Campaign', min_value=1)
    pdays = st.number_input('Days Passed Since Last Contact', min_value=-1)
    previous = st.number_input('Number of Contacts Before this Campaign')
    poutcome = st.selectbox('Previous Campaign Outcome', ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.slider("Employee Variation Rate", min_value=-3.4, max_value=1.4, value=0.0, step=0.1)
    cons_price_idx = st.slider("Consumer Price Index", min_value=92.0, max_value=95.0, step=0.01)
    cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-51.0, max_value=-27.0, step=0.1)
    euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.63, max_value=5.05, step=0.1)
    nr_employed = st.slider("Select nr_employed", min_value=4964, max_value=5228, step=2)

    # Create a list with user input
    input_data = [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
    
    # Label encoding for categorical features
    job_mapping = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11}
    marital_mapping = {'married': 0, 'single': 1, 'divorced': 2}
    education_mapping = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 7}
    default_mapping = {'no': 0, 'yes': 1}
    housing_mapping = {'no': 0, 'yes': 1}
    loan_mapping = {'no': 0, 'yes': 1}
    contact_mapping = {'cellular': 0, 'telephone': 1}
    month_mapping = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
    day_of_week_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
    poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2}
    
    input_data[1] = job_mapping[input_data[1]]
    input_data[2] = marital_mapping[input_data[2]]
    input_data[3] = education_mapping[input_data[3]]
    input_data[4] = default_mapping[input_data[4]]
    input_data[5] = housing_mapping[input_data[5]]
    input_data[6] = loan_mapping[input_data[6]]
    input_data[7] = contact_mapping[input_data[7]]
    input_data[8] = month_mapping[input_data[8]]
    input_data[9] = day_of_week_mapping[input_data[9]]
    input_data[14] = poutcome_mapping[input_data[14]]
    
    # Code for prediction
    if st.button('Predict'):
        result = prediction(input_data)
        st.success(result)

if __name__ == '__main__':
    main()
