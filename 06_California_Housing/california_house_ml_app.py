# import libraries
import streamlit as st
import pandas as pd
import shap # for SHAP values
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# App title
st.title('California House Price Prediction App')

# sub-title
st.write("""
### App to predict the california House Price
This app predicts the **california House Price**!
""")

# load the california dataset
california = datasets.fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
Y = pd.DataFrame(california.target, columns=["MEDV"])

# sidebar
# header of sidebar
st.sidebar.header('Specify Input Parameters')

# function to get user input
def user_input_features():
    medinc = st.sidebar.slider('MedInc', float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean())),
    houseage = st.sidebar.slider('HouseAge', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean())),
    aveRooms = st.sidebar.slider('AveRooms', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean())),
    aveBedrms = st.sidebar.slider('AveBedrms', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean())),
    population = st.sidebar.slider('Population', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean())),
    aveOccup = st.sidebar.slider('AveOccup', float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean())),
    latitude = st.sidebar.slider('Latitude', float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean())),
    longitude = st.sidebar.slider('Longitude', float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))

    data = {'MedInc': medinc,
            'HouseAge': houseage,
            'AveRooms': aveRooms,
            'AveBedrms': aveBedrms,
            'Population': population,
            'AveOccup': aveOccup,
            'Latitude': latitude,
            'Longitude': longitude}
    
    features = pd.DataFrame(data, index=[0])
    return features
    
df = user_input_features()


# main panel

# print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# build regression model
model = RandomForestRegressor()
model.fit(X, Y)

# apply model to make predictions
prediction = model.predict(df)

# print prediction
st.header('Prediction of MEDV')
st.write("<span style='color: black; font-size:20px'>"+[prediction][0] +"</span>", unsafe_allow_html=True)
st.write('---')

# Explaining the model's predictions using SHAP values
explain_model = shap.TreeExplainer(model)
shap_values = explain_model.shap_values(X)

# Feature importance
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

