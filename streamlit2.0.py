import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RandomForestClassifier
from sklearn.metrics import r2_score

df=pd.read_csv("bank-additional-full.csv")

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("ce dataset provient d'une campagne de marketing direct menée par une institution bancaire portugaise,principalement par le biais d'appel téléphonique.et à pour objectif de prédire si un client souscrira à un dépot à terme(variable cible y='oui' ou 'non'")
    
    
    st.write("cette visualisation à été realiser par DANG-NE DJEMO MIGUEL IVAN")
    
    st.image("")
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    
   
        
elif page == pages[2]:
    st.write("### Analyse de données")
    
    fig = sns.displot(x='price', data=df, kde=True)
    plt.title("Distribution de la variable cible price")
    st.pyplot(fig)
    
    fig2 = px.scatter(df, x="age", y="area", title="Evolution du prix en fonction de la surface")
    st.plotly_chart(fig2)
    
    fig3, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    plt.title("Matrice de corrélation des variables du dataframe")
    st.write(fig3)

elif page == pages[3]:
    st.write("### Modélisation")
    
    df_prep = pd.read_csv("bank_additional_full.csv")
    
    y = df_prep["y"]
    X= df_prep.drop("y", axis=1)
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=123)
    
    scaler = StandardScaler()
    num = ['age', 'euribor3m', 'emp.var.ratems', 'cons.price.idx',]
    X_train[num] = scaler.fit_transform(X_train[num])
    X_test[num] = scaler.transform(X_test[num])
    
   
    
    y_pred_reg=reg.predict(X_test)
    y_pred_rf=rf.predict(X_test)
    y_pred_knn=knn.predict(X_test)
    
    model_choisi = st.selectbox(label = "Modèle", options = ['Random Forest'])
    
    def train_model(model_choisi) : 
    model_choisi == 'Random Forest'
           
    r2 = r2_score(y_test, y_pred)
    return r2
    
    st.write("Coefficient de détermination", train_model(model_choisi))