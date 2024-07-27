import streamlit as st
import joblib 
import pandas as pd
import numpy as np

heart_attack_label = {0: 'No Risk of Heart Attack', 1: 'At Risk of Heart Attack '}


st.write("""
# Simple Heart Attack Prediction App
This app predicts if user will have Heart Attack!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    diabetes = st.sidebar.selectbox('Have Diabetes?', options=["Yes", "No"])
    family_history = st.sidebar.selectbox('Is there family history of heart attack?', options=["Yes","No"])
    smoking = st.sidebar.selectbox('Do you smoke?', options=["Yes","No"])
    alcohol = st.sidebar.selectbox('Do you frequently consume alcohol?', options=["Yes","No"])
    prev_heart_prob = st.sidebar.selectbox('Do you have previous heart problems?', options=["Yes","No"])
    medication_use = st.sidebar.selectbox('Are you on medication?', options=["Yes","No"])
    physical_activity = st.sidebar.slider('How many days in a week do u excercise?',0,7,3)
    sleep = st.sidebar.slider('How many hours do you sleep a day?', 4,10,5)
    age_map = st.sidebar.selectbox('Age Group?', options=['Child(0-17y/o)', 'Young Adult(18-34y/o)', 'Adult(35-49y/o)', 'Senior(50-64y/o)', 'Elderly(>65y/o)'])
    cholesterol_map = st.sidebar.selectbox('Cholesterol Level?', options=['Healthy(0-199)', 'At-Risk(200-238)', 'Dangerous(>239)'])
    bmi_map = st.sidebar.selectbox('Body Mass Index Level (BMI)?', options=['Underweight(0-18)', 'Normal(19-24)', 'Overweight(25-29)', 'Obese(>30)'])
    blood_pressure_map = st.sidebar.selectbox('Systolic Blood Pressure Level?', options=['Normal(0-119)', 'Elevated(120-129)', 'High Type 1(130-139)', 'High Type 2(140-179)', 'Hypertension Crisis(>180)'])
    diet_map=st.sidebar.selectbox('Overall Diet Plan?', options=['Unhealthy','Average','Healthy'])
    data = {'Diabetes': diabetes,
            'Family History': family_history,
            'Smoking': smoking,
            'Alcohol Consumption': alcohol,
            'Previous Heart Problems': prev_heart_prob,
            'Medication Use': medication_use,
            'Physical Activity Days Per Week': physical_activity,
            'Sleep Hours Per Day': sleep,
            'age_mapped': age_map,
            'bmi_mapped': bmi_map,
            'cholesterol_mapped': cholesterol_map,
            'systolic_blood_pressure_mapped': blood_pressure_map,
            'diet_mapped': diet_map,}
    
    #Converting user input to numerical values
    if data['Diabetes'] == 'Yes':
        data['Diabetes'] = 1
    else:
        data['Diabetes'] = 0

    if data['Family History'] == 'Yes':
        data['Family History'] = 1 
    else:
       data['Family History'] = 0

    if data['Smoking'] == 'Yes':
        data['Smoking'] = 1 
    else:
        data['Smoking'] = 0

    if data['Alcohol Consumption'] == 'Yes':
        data['Alcohol Consumption'] = 1 
    else:
        data['Alcohol Consumption'] = 0

    if data['Previous Heart Problems'] == 'Yes':
        data['Previous Heart Problems'] = 1 
    else: 
        data['Previous Heart Problems'] = 0

    if data['Medication Use'] == 'Yes':
        data['Medication Use'] = 1 
    else:
        data['Medication Use'] = 0
    
    age_mapping = {'Child(0-17y/o)': 1, 'Young Adult(18-34y/o)': 2, 'Adult(35-49y/o)': 3, 'Senior(50-64y/o)': 4, 'Elderly(>65y/o)': 5}
    cholesterol_mapping = {'Healthy(0-199)': 1, 'At-Risk(200-238)': 3, 'Dangerous(>239)': 4}
    bmi_mapping = {'Underweight(0-18)': 1, 'Normal(19-24)': 2, 'Overweight(25-29)': 3, 'Obese(>30)': 4}
    blood_pressure_mapping = {'Normal(0-119)': 1, 'Elevated(120-129)': 2, 'High Type 1(130-139)': 3, 'High Type 2(140-179)': 4, 'Hypertension Crisis(>180)': 5}
    diet_mapping = {'Unhealthy': 1, 'Average': 2, 'Healthy': 3}
    
    data['age_mapped'] = age_mapping[data['age_mapped']]
    data['cholesterol_mapped'] = cholesterol_mapping[data['cholesterol_mapped']]
    data['bmi_mapped'] = bmi_mapping[data['bmi_mapped']]
    data['systolic_blood_pressure_mapped'] = blood_pressure_mapping[data['systolic_blood_pressure_mapped']]
    data['diet_mapped'] = diet_mapping[data['diet_mapped']]

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)


model=joblib.load('XGB_Classifier.pkl')


prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)