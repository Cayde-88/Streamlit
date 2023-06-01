import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def app():
    st.title('Model')
    st.write('This is the `Model` page of the multi-page app.')
    st.write('The model performance of different classifiers are as follows:')

    st.sidebar.header('User Input Parameters')

    #  Create input parameters
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)  # (label, min, max, default)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])  # index=[0] to make it a dataframe
        return features

    #  Store the input parameters into a dataframe
    df = user_input_features()

    # create a vertical space
    st.write('---')
    st.write('\n\n\n\n')

    # Display the input parameters
    st.subheader('User Input parameters')
    st.write(df)
    # new line
    st.write('---')

    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Train the model
    rfc = RandomForestClassifier()
    rfc.fit(X, Y)

    # Predict the input parameters
    prediction = rfc.predict(df)
    prediction_proba = rfc.predict_proba(df)

    # Display the prediction
    st.subheader('Prediction')
    st.write(iris.target_names[prediction][0],
            'with a probability of',
            prediction_proba.max(),
            unsafe_allow_html=True,
            style={'font-size': '200px'})

    # new line
    st.write('---')

    # Display the prediction probability in a table
    st.subheader('Prediction Probability in a Table')
    st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))