import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained Logistic Regression model
model = joblib.load(open('breast_cancer_model.pkl', 'rb'))

# Load the CSV file (patient data)
@st.cache_data
def load_data():
    # Assuming the CSV file is named 'breast_cancer.csv'
    return pd.read_csv('breast_cancer.csv')

# Load the patient data into a DataFrame
patient_data = load_data()

# Drop any irrelevant or missing columns (e.g., 'Unnamed: 32')
if 'Unnamed: 32' in patient_data.columns:
    patient_data = patient_data.drop(columns=['Unnamed: 32'])

# Title of the app
st.title("🏥 Breast Cancer Prediction App")
st.markdown("### Predict the nature of the tumor based on patient data.")

# Input patient ID
patient_id = st.text_input("🔍 Enter Patient ID", '')

# Check if the ID is valid and retrieve patient data
if patient_id in patient_data['id'].astype(str).values:
    # Get the row corresponding to the entered patient ID
    user_row = patient_data[patient_data['id'].astype(str) == patient_id]

    # Extract the 30 feature values (excluding 'id' and 'diagnosis')
    patient_parameters = user_row.drop(columns=['id', 'diagnosis']).values.flatten()

    # Display the 30 parameters in a more visually appealing format
    st.subheader(f"Patient ID: **{patient_id}** recognized.")
    
    # Create columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # Patient parameters
    with col1:
        st.write("### Patient Parameters:")
        st.write(user_row.drop(columns=['id', 'diagnosis']).T.rename(columns={0: 'Value'}))

    # Convert the patient parameters into the right format for the model (reshape to 2D)
    patient_parameters = np.array(patient_parameters).reshape(1, -1)  # Reshape to 2D

    # Automatically predict once the ID is valid and parameters are loaded
    prediction = model.predict(patient_parameters)

    # Display prediction result
    with col2:
        if prediction[0] == 0:
            st.success("✅ The tumor is classified as **Malignant**.")
        else:
            st.success("✅ The tumor is classified as **Benign**.")
else:
    if patient_id != '':
        st.warning("⚠️ Invalid Patient ID! Please enter a valid ID.")
