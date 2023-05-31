# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes, load_wine

# Page layout
st.set_page_config(page_title = "Machine Learning Hyperparameter Optimization App",layout="wide")

# sub-header
st.write("""
# Machine Learning Hyperparameter Optimization App
**This app performs the following:**
* Hyperparameter Optimization
* Cross Validation
* Grid Search
""")

# sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader('Upload your input CSV file',type=['csv'])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

with st.sidebar.header('Set Parameters'):
    split_size = st.sidebar.slider("Data split ratio (% for Training Set)", 10, 90, 80, 5)

    
with st.sidebar.subheader("Learning Parameters"):
    parameter_n_estimators = st.sidebar.slider("Number of estimators (n_estimators)", 0, 500, (10,50), 50)
    parameter_n_estimators_step = st.sidebar.number_input("Step size for n_estimators", 10)
    st.sidebar.write('---')
    parameter_max_features = st.sidebar.slider("Max features (max_features)", 1, 50, (1,3), 1)
    st.sidebar.number_input("Step size for max_features", 1)
    st.sidebar.write('---')
    parameter_min_samples_split = st.sidebar.slider("Minimum number of samples required to split an internal node (min_samples_split)", 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider("Minimum number of samples required to be at a leaf node (min_samples_leaf)", 1, 10, 2, 1)

with st.sidebar.subheader('General Parameters'):
    parameter_learning_rate = st.sidebar.slider("Learning rate (learning_rate)", 0.01, 1.0, 0.1, 0.01)
    parameter_random_state = st.sidebar.slider("Seed number (random_state)", 0, 1000, 42, 1)
    parameter_criteria = st.sidebar.select_slider("Performance measure (criteria)", options=["absolute_error", "squared_error"])
    parameter_n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs)", options=[1, -1])

    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)


# main
st.subheader("Dataset")

# model building
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href

def build_model(df):
    X = df.iloc[:,:-1] # using all column except for the last column as X
    Y = df.iloc[:,-1] # select the last column as Y

    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)

    # build model
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                random_state=parameter_random_state,
                                max_features=parameter_max_features,
                                criterion=parameter_criteria,
                                min_samples_split=parameter_min_samples_split,
                                min_samples_leaf=parameter_min_samples_leaf,
                                n_jobs=parameter_n_jobs)
    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(X_train, Y_train)

    st.subheader('2. Model Performance')
    Y_pred_test = grid.predict(X_test)

    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.write("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    
    st.subheader('3. Model Parameters')
    st.write(grid.get_params())

    #-----Process Grid Search Result-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)

    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()

    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    #-----Plot-----#
    layout = go.Layout(
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text='n_estimators')
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text='max_features')
            ) )
    
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning',
                    scene = dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='R2'),
                    autosize=False,
                    width=800, height=800,
                    margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    #-----Save Grid Search Result-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x,y,z], axis=1)
    st.markdown(filedownload(grid_results), unsafe_allow_html=True)

# dataset selection
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df)
else:
    st.info("Awaiting for CSV file to be uploaded.")
    if st.button("Press to use Example Dataset"):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)