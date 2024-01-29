import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.header("Welcome to Mumbai House Price Prediction" ,divider="violet")

pipe1 = pickle.load(open("Prediction_Pipeline.pkl" ,"rb"))
df = pickle.load(open("House_DataFrame.pkl","rb"))

region = st.selectbox(
    'Select Region In Which You Want to Buy House',
    (df["region"].unique()),
    index=None,
    placeholder="Select Region...")

house_type = st.selectbox(
    "Select House Type ",
    (df["type"].unique()),
    index=None,
    placeholder="Select House Type...")

BHK = st.selectbox(
    "Select How Many BHKs House You want",
    (df["bhk"].unique()),
    index=None,
    placeholder="Select BHK...")

sqft = st.number_input("Enter House Square Feet", value=None, placeholder="Enter Square Feet...")

L = [region ,house_type ,BHK ,sqft]

if st.button('Predict'):
    
    inp = pd.DataFrame(np.array(L).reshape(1,4),columns=['region', 'type', 'bhk', 'area'])
    Predict = pipe1.predict(inp)
    st.header("",divider='violet')
    st.write("House Details : ")
    st.subheader(f"{region} ,{house_type} ,{BHK}BHK ,{sqft}sqft")
    st.header("",divider='rainbow')
    st.write("Predicted Price by Model is: ")
    st.header(f"{Predict[0]} ₹")
    st.header("",divider='violet')


# inp = pd.DataFrame(np.array(["Kanjurmarg" ,"Apartment" ,2 ,730]).reshape(1,4),columns=['region', 'type', 'bhk', 'area'])

# st.write(inp)
# Predict = pipe1.predict(inp)

# st.write(predict.astype(int))


