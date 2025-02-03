import pandas as pd
import numpy as np
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RandomForestClassifier
import joblib 
from sklearn.metrics import r2_score
df=pd.read_csv('bank_additioal_full.csv')
st.dataframe(df.head())
st.sidebar.title("sommaire")
pages = ["Contexte du projet","Exploration des données","analyse des données","Modélisation"]
page= st.sidebar.radio("Aller vers la page",pages)
if page == page[0]
    st.write("### A propos")
    st.write("ce dataset provient d'une campagne de marketing direct menée par une institution bancaire portugaise,principalement par le biais d'appel téléphonique.et à pour objectif de prédire si un client souscrira à un dépot à terme(variable cible y='oui' ou 'non'")
    st.write("cette visualisation à été realiser par DANG-NE DJEMO MIGUEL IVAN")
 elif page == page[1]
    st.write("exploration des données")
    st.dataframe(df.head())
    st.write("dimension du dataframe:")
    st.write(df.shape)
elif page == page[2]
    st.write("### Analyse des données")
    fig=plt.figure(figsize=(8,5))
    sns.heatmap(corr,annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig)
    g=df.groupby(["y","marital"])["marital"].count().transform(lambda x:x/x.sum())
    g= g.unstack(level=0)
    fig=g.plot(kind="bar",figsize=(9,8),use_index=True,)
elif page == page[3]
    st.write("### Modélisation")
    df_encoder= pd.read_csv("bank_additional_encoded.csv")
    y= df_encoder.prep("y")
    x= df_encoder.drop("y",axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
    scaler=StandardScaler()
    