from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import preprocessing
import streamlit as st

# Streamlit
st.sidebar.header("Deploying Decision Tree Classifier")

def user_input_features():
    SepalLength = st.sidebar.slider('Sepal Length', 4.0, 10.0, 6.0)
    SepalWidth = st.sidebar.slider('Sepal Width', 2.0, 4.5, 3.0)
    PetalLength = st.sidebar.slider('Petal Length', 1.0, 7.0, 3.5)
    PetalWidth = st.sidebar.slider('Petal Width', 0.1, 2.5, 1.3)
    
    data = {
        'Sepal.Length': SepalLength,
        'Sepal.Width': SepalWidth,
        'Petal.Length': PetalLength,
        'Petal.Width': PetalWidth
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Taking user input
df = user_input_features()
st.subheader('User Parameters')
st.write(df)

# Import the CSV file for training
iris = pd.read_csv("iris.csv")
if "Unnamed: 0" in iris.columns:
    iris = iris.drop(columns=["Unnamed: 0"])
st.subheader("Dataset Preview")
st.dataframe(iris.head())

# Label encoding
label_encoder = preprocessing.LabelEncoder()
iris['Species'] = label_encoder.fit_transform(iris['Species'])

# Training data
x = iris.drop(columns=['Species'], axis=1)
y = iris['Species']

# Model training
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
model_gini.fit(x, y)

# Predict
prediction = model_gini.predict(df)  # Store the prediction
predicted_species = label_encoder.inverse_transform(prediction)  # Convert back to species name

# Display prediction
st.subheader('Prediction')
st.write(f"Predicted Species: **{predicted_species[0]}**")