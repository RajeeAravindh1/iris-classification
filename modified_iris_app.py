import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/iris-species.csv")
df["Species"]=df["Species"].map({"Iris-setosa":0, 'Iris-virginica': 1, 'Iris-versicolor':2})

features=df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
target=df["Species"]
X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.30,random_state=42)

svc=SVC(kernel="linear").fit(X_train,y_train)
log_reg=LogisticRegression(n_jobs=-1).fit(X_train,y_train)
rfc=RandomForestClassifier(n_jobs=-1,n_estimators=100).fit(X_train,y_train)

def prediction(model,sepal_length,sepal_width,petal_length,petal_width):
	predicted_value=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]
	if predicted_value==0:
		return "Iris-setosa"
	if predicted_value==1:
		return "Iris-virginica"
	if predicted_value==2:
		return "Iris-versicolor"

st.sidebar.title("Iris Flower Classification ")
sepal_length=st.sidebar.slider("Sepal Length",0.0,10.0)
sepal_width=st.sidebar.slider("Sepal Width",0.0,10.0)
petal_length=st.sidebar.slider("Petal Length",0.0,10.0)
petal_width=st.sidebar.slider("Petal Width",0.0,10.0)

classifier=st.sidebar.selectbox("Classifier",("Random Forest Classifier","Logisitic Regression","Support Vector Machine"))
if st.sidebar.button("Predict"):
	if classifier=="Random Forest Classifier":
		predict=prediction(rfc,sepal_length,sepal_width,petal_length,petal_width)
		score=rfc.score(X_train,y_train)

	elif classifier=="Logisitic Regression":
		predict=prediction(log_reg,sepal_length,sepal_width,petal_length,petal_width)
		score=log_reg.score(X_train,y_train)

	else:
		predict=prediction(svc,sepal_length,sepal_width,petal_length,petal_width)
		score=svc.score(X_train,y_train)


	st.write("Species Predicted by the model:",predict)
	st.write("Accuracy of the model:",score)










