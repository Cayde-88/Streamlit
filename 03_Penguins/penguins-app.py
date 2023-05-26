import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# background image
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: fill;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


# header
st.write("""
# Penguin Prediction App
### This app predicts the **Palmer Penguin** species!
""")


# sidebar
st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island, 
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# combine user input with penguins dataset
penguins_raw = pd.read_csv('./model-building/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# encode
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] # select only the first row (the user input data)

# display user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)

else:
    st.write('<span class="my-text">Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).</span>', unsafe_allow_html=True)
    st.write(df)

# load the saved model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Species prediction
st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write("<span style='color: black;'>"+penguins_species[prediction][0] +"</span>", unsafe_allow_html=True)

# Probability
st.subheader('Prediction Probability')
st.write(prediction_proba)

# Set font color
st.markdown("<style>h1{color: #000000;}</style>", unsafe_allow_html=True)
st.markdown("<style>h3{color: #000000;}</style>", unsafe_allow_html=True)
st.markdown("<style>.my-text{color: #000000;}</style>", unsafe_allow_html=True)

# set background
set_bg_hack("bg.jpg")

# comment