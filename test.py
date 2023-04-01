
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# give a title to our app
st.title('Welcome to Iris flower Analysis')
# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
# Display the dataset
st.write("## Iris Dataset")
st.write("Number of samples:", X.shape[0])
st.write("Number of features:", X.shape[1])
st.write("Number of classes:", len(class_names))
# Show a table of the data
data_table = pd.DataFrame(X, columns=iris.feature_names)
data_table['target'] = y
data_table['target'] = data_table['target'].map({i: class_names[i] for i in range(len(class_names))})
st.write(data_table)
# Set up a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model to the data
rfc.fit(X, y)

# Add input fields for sepal length, sepal width, petal length, and petal width
st.sidebar.write("## Input Features")
sepal_length = st.sidebar.slider("Sepal length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Display the selected input values
st.write("## Selected Input Values")
st.write("Sepal length:", sepal_length)
st.write("Sepal width:", sepal_width)
st.write("Petal length:", petal_length)
st.write("Petal width:", petal_width)
# Define a prediction button that takes in the input values and predicts the type of iris flower
if st.sidebar.button("Predict"):
    # Make a prediction using the input values and the classifier
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = rfc.predict(input_data)

    # Display the predicted class name
    st.write("## Prediction")
    st.write(class_names[prediction[0]])

