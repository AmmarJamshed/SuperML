#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Define a function to load the dataset
def load_data():
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    else:
        return None


# In[3]:


# Define a function to preprocess the dataset
def preprocess_data(df, exclude_columns):
    df = df.drop(columns=exclude_columns)
    df = df.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
    return df


# In[7]:


# Define Model training
def train_model(df, target_variable, model_choice, test_size):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_choice == 'Random Forest Classifier':
        model = RandomForestClassifier()
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression()
    elif model_choice == 'Random Forest Regressor':
        model = RandomForestRegressor()
    elif model_choice == 'Linear Regression':
        model = LinearRegression()
    else:
        st.error("Unsupported model choice")
        return None

        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_test, y_test, predictions


# In[8]:


# Define a function to evaluate the model
def evaluate_model(model, y_test, predictions):
    if model._estimator_type == 'classifier':
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, predictions)
        st.write(f"Model Mean Squared Error: {mse:.2f}")


# In[9]:


# Define a function to download the predictions
def download_predictions(df, X_test, predictions, target_variable):
    X_test[target_variable + '_predictions'] = predictions
    st.write(X_test.head())
    csv = X_test.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Predictions as CSV",
        csv,
        "predictions.csv",
        "text/csv",
        key='download-csv'
    )


# # Streamlit deply

# In[10]:


# Streamlit app layout
st.title("Supervised Machine Learning Tool")
df = load_data()

if df is not None:
    st.write("Data Preview:", df.head())

    exclude_columns = st.multiselect("Select columns to exclude", options=df.columns.tolist())
    df = preprocess_data(df, exclude_columns)

    target_variable = st.selectbox("Select Target Variable", options=df.columns.tolist())
    model_choice = st.selectbox("Select Model", ['Random Forest Classifier', 'Logistic Regression', 'Random Forest Regressor', 'Linear Regression'])
    test_size = st.slider("Select Test Size (for Train/Test Split)", min_value=0.1, max_value=0.5, value=0.3)

    if st.button("Train Model"):
        model, X_test, y_test, predictions = train_model(df, target_variable, model_choice, test_size)
        if model is not None:
            evaluate_model(model, y_test, predictions)
            download_predictions(df, X_test, predictions, target_variable)


# In[ ]:




