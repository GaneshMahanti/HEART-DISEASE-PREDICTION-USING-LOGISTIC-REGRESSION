# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:44:47 2023

@author: shash
"""

import numpy as np 
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('D:/mini_project/deployment/trained_model.sav', 'rb'))
def hprediction(input_data):
    input_data = (60,0,0,140,268,0,0,160,0,3.6,0,2,2)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
      return 'The Person does not have a Heart Disease'
    else:
      return 'The Person has Heart Disease'
 
    
def main():
    
    #giving title
    st.title('Heart Disease Prediction Web App')
    
    #assigning input variables
    age=st.text_input('age')
    sex=st.text_input('sex[0-Male,1-Female]')
    cp=st.text_input('chest pain type[4 values(0,1,2,3)]')
    trestbps=st.text_input('rest bp range[60-200]top number')
    chol=st.text_input('serum cholestoral in mg/dl-value')
    fbs=st.text_input('fasting blood sugar > 120 mg/dl[0-no,1-yes]')
    restecg=st.text_input('resting electrocardiographic results (values 0,1,2)')
    thalach=st.text_input('maximum heart rate achieved-value')
    exang=st.text_input('exercise induced angina values[0-no,1-yes]')
    oldpeak=st.text_input('oldpeak = ST depression induced by exercise relative to rest values[0-9]')
    slope=st.text_input('the slope of the peak exercise ST segment values[0-9]')
    ca=st.text_input('number of major vessels [0-3] colored by flourosopy')
    thal=st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    #prediction
    diagnosis=''
    
    #creating a button
    if st.button('submit'):
        diagnosis=hprediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
        
    st.success(diagnosis)
    
 
#this is because the function should run only when rum through anaconda but not when we import as file    
if __name__=='__main__':
    main()
    
    
    
    