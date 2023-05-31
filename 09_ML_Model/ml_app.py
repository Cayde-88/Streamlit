# import libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_wine

## page layout
## page expands to full width
st.set_page_config(page_title='The Machine Learning App',layout='wide')

# model building
def build_model(df):
    X = df.iloc[:,:-1] # select all columns except the last column
    Y = df.iloc[:,-1] # select the last column as target

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X.shape)
    st.write('Test set')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    # data split
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=param_split_size)

    random_forest = RandomForestRegressor(
        n_estimators=param_n_estimators,
        max_features=param_max_features,
        criterion=param_criterion,
        max_depth=param_max_depth,
        bootstrap=param_bootstrap,
        min_samples_leaf=param_min_samples_leaf,
        min_samples_split=param_min_samples_split,
        oob_score=param_oob_score,
        n_jobs=params_n_jobs,
        random_state=param_random_state
    )

    random_forest.fit(X_train,Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = random_forest.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train,Y_pred_train))
    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train,Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = random_forest.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test,Y_pred_test))
    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test,Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(random_forest.get_params())

# main
st.title('The Machine Learning App')

## sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader('Upload your input CSV file',type=['csv'])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

with st.sidebar.header('2. Set Parameters'):
    param_split_size = st.sidebar.slider('Data split ratio (% for Training Set)',10,90,80,5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    param_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)',0,1000,100,100)
    param_max_features = st.sidebar.select_slider('Max features (max_features)',options=['auto','sqrt','log2'])
    param_min_samples_leaf = st.sidebar.slider('Minimum number of samples (min_samples_leaf)',1,10,2,1)
    param_min_samples_split = st.sidebar.slider('Minimum number of samples (min_samples_split)',2,10,2,1)
    param_max_depth = st.sidebar.select_slider('Max depth (max_depth)',options=[None,1,2,3,4,5,6,7,8,9,10])

with st.sidebar.subheader('2.2. General Parameters'):
    param_random_state = st.sidebar.slider('Seed number (random_state)',0,1000,42,1)
    param_criterion = st.sidebar.select_slider('Performance measure (criterion)',options=['squared_error','absolute_error'])
    param_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)',options=[True,False])
    param_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)',options=[False,True])
    params_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)',options=[1,-1])

## main panel
# displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)

else: 
    st.info('Awaiting for CSV file to be uploaded.')
    st.write("Example datasets are available.")
        
    col1,col2, col3, col4, col5, col6 = st.columns(6) # col 3 - 6 are empty

    # wine dataset
    if col1.button("Wine Dataset"):        
        wine = load_wine()
        X = pd.DataFrame(wine.data,columns=wine.feature_names)
        Y = pd.Series(wine.target,name='target')
        df_wine = pd.concat( [X,Y],axis=1 )

        st.markdown('The **Wine** dataset is used as the example.')
        st.write(df_wine.head(5))

        build_model(df_wine)

    if col2.button("Diabetes Dataset"):
        # diabetes dataset
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target,name='response')
        df_diab = pd.concat( [X,Y],axis=1 )

        st.markdown('The **Diabetes** dataset is used as the example.')
        st.write(df_diab.head(5))
        
        build_model(df_diab)




