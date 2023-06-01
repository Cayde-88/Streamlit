# import libaries
import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
import requests
from bs4 import BeautifulSoup

# Title
st.markdown('''
# Stock Price App
Shown are the stock **closing price** and ***volume*** of queryied stock!
''')

st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2023, 5, 31))

# Retrieving tickers data
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) # Get the historical prices for this ticker

# Ticker information
# get long name
string_name = tickerData.info['longName']

# get logo
url = f'https://en.wikipedia.org/wiki/{string_name.replace(" ", "_")}'  # Replace spaces with underscores
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
logo_element = soup.find('td', class_='infobox-image logo')

if logo_element:
    string_logo = '<img src=%s>' % logo_element.find('img')['src']
    st.markdown(string_logo, unsafe_allow_html=True)

st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

# Ticker data
st.header('**Ticker data**')
st.write(tickerDf)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)
