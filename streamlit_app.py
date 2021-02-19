# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:19:11 2021

@author: ASUS
"""

#%%
###import packages


import streamlit as st
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time








#%%

# Text/Title
st.title("Logistic Regression")

#%%
#Navigation bar

st.sidebar.header('Dataset and Hyperparameters')

st.sidebar.markdown("""**Select Dataset**""")
Dataset = st.sidebar.selectbox('Dataset',('Iris','Wine',
                                          'Breast Concer'))

Split = st.sidebar.slider('Train-Test Splitup (in %)', 0,100,70)
st.sidebar.markdown("""**Select Logistic Regression Parameters**""")

Solver= st.sidebar.selectbox('Algorithm to optimise',('lbfgs','newton-cg', 'liblinear', 'sag', 'saga'))
Penality = st.sidebar.radio("Penality:", ('none','l1', 'l2','elasticnet'))
Tol= st.sidebar.text_input("Tolerance for stopping Criteria (default: 1e-4)","1e-4")
Max_Iteration=st.sidebar.text_input("Number of iteration (default: 50)","50")


#%%
#creating dictionary for hyperparameters
parameters={ 'Penality':Penality,
             'Tol':Tol,
             'Max_Iteration':Max_Iteration,
             'Solver':Solver}
       

#%%

#%%

#Functions
#Function for the dataset display
def load_dataset(dataset):
    if dataset == 'Iris':
        data=sklearn.datasets.load_iris()
    elif dataset == 'Wine':
         data=sklearn.datasets.load_wine()
    elif dataset == 'Breast Concer':
         data=sklearn.datasets.load_breast_cancer()
    
    return data


#test-train split
def train_test_splited(data,split):
    X_train, X_test, y_train, y_test = train_test_split(data.data,data.target, test_size=split,
    random_state=42)

    return ( X_train, X_test, y_train, y_test)

#model for Logic regression    

#%%

#Body

  #Dataset Information
st.markdown("""***Dataset Information***""")
st.write("*Dataset Name* :",Dataset)
Data=load_dataset(Dataset)
targets=Data.target_names
Dataframe=pd.DataFrame (Data.data,columns=Data.feature_names)
Dataframe['target']=pd.Series(Data.target)
Dataframe['target labels']=pd.Series(targets[i] for i in Data.target)
st.write("*Dataset Feature Overview*")
st.write(Dataframe)


#train Model navbar button
if(st.sidebar.button("Click to train the Logistic Regression Model")):
    
    #model Information
    with st.spinner('Loading Dataset...'):
        time.sleep(1)
    st.success("Dataset Loaded")
    
    
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    with st.spinner('Trainning...'):
        time.sleep(1)
    st.success("Model Trained") 

    

    


   














    






