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
    family_history = st.sidebar.selectbox('Past Family History?', options=["Yes","No"])
    smoking = st.sidebar.selection('Do you smoke?', options=["Yes","No"])
    alcohol = st.sidebar.selection('Do you frequently consume alcohol?', options=["Yes","No"])
    prev_heart_prob = st.sidebar.selection('Do you have previous heart problems?', options=["Yes","No"])
    medication_use = st.sidebar.selection('Are you on medication?', options=["Yes","No"])
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