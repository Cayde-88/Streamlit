# import libraries
import streamlit as st
from multiapp import MultiApp
from apps import home, data, model

# Create an instance of the app
app = MultiApp()

# title of the page
st.title('Multi-page app')

# Add all your application here
app.add_app('Home', home.app)
app.add_app('Data', data.app)
app.add_app('Model', model.app)

# The main app
app.run()