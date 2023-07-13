# import libraries
import streamlit as st
import pandas as pd
import zipfile
import base64
import os

# set title
st.title('Spreadsheet File Combiner')

# Excel file merger
def excel_merger(zip_file_name):
    df = pd.DataFrame()
    archive = zipfile.ZipFile(zip_file_name, 'r')
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        for file in zip_ref.namelist():
            xlfile = archive.open(file)
            if file.endswith('.xlsx') or file.endswith('.csv'):
                df_xl = pd.read_excel(xlfile)
                df_xl['Notes'] = file
                df = df.append(df_xl, ignore_index=True)
    return df

# upload file
with st.sidebar.header('1. Upload your zip file'):
    uploaded_file = st.sidebar.file_uploader("Upload your input zip file", type=["zip"])
    st.sidebar.markdown("""
    [Example input file](https://raw.githubusercontent.com/stevekm/Streamlit-Apps/master/16_Combine_CSV/Example.zip)
    """)


# download file
def file_downloader(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
    return href

def excel_downloader(df):
    df.to_excel('output.xlsx', index=False)
    data = open('output.xlsx', 'rb').read()
    b64 = base64.b64encode(data).decode('UTF-8')
    href = f'<a href="data:file/xlsx;base64,{b64}" download="output.xlsx">Download Excel File</a>'
    return href

# main
if st.sidebar.button("Submit"):
    df = excel_merger(uploaded_file)
    st.header('2. Output')
    st.write(df)
    st.markdown(file_downloader(df), unsafe_allow_html=True)
    st.markdown(excel_downloader(df), unsafe_allow_html=True)
    os.remove('output.xlsx')

else:
    st.info('Upload zip file and click Submit.')
