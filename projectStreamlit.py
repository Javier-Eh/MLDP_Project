import streamlit as st
import joblib 
import pandas as pd
import numpy as np

model=joblib.load('XGB_Classifier.pkl')
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
    physical_activity = st.sidebar.slider('How many days in a week do u excercise?',0,7,1)
    sleep = st.sidebar.slider('How many hours do you sleep a day?', 4,10,1)
    age_map = st.sidebar.selectbox('Age Group?', options=['Child (0-17y/o)', 'Young Adult(18-34y/o)', 'Adult(35-49y/o)', 'Senior(50-64y/o)', 'Elderly(>65y/o)'])
    cholesterol_map = st.sidebar.selectbox('Cholesterol Level?', options=['Healthy(0-199)', 'At-Risk(200-238)', 'Dangerous(>239)'])
    bmi_map = st.sidebar.selectbox('Body Mass Index Level (BMI)?', options=['Underweight(0-18)', 'Normal(19-24)', 'Overweight(25-29)', 'Obese(>30)'])
    blood_pressure_map = st.sidebar.selectbox('Systolic Blood Pressure Level?', options=['Normal(0-119)', 'Elevated(120-129)', 'High Type 1(130-139)', 'High Type 2(140-179)', 'Hypertension Cisis(>180)'])
    diet_map=st.sidebar.selectbox('Overall Diet Plan?', options=['Unhealthy','Average','Healthy'])
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)