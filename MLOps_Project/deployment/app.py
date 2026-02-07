# Script for the Streamlit UI app

# Importing the necessary libraries
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="KavinPrasathK/Tourism_Package_Prediction", filename="best_tourism_package_prediction_model_v1.joblib")
    model = joblib.load(model_path)
    return model

model = load_model()

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package.
Please enter the customer details and interaction data below to get a prediction.
""")

# User input fields
st.header("Customer Details")
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Large Business'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
NumberOfTrips = st.number_input("Number of Trips (Annually)", min_value=0, max_value=50, value=1)
Passport = st.radio("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
OwnCar = st.radio("Owns Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5 years)", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)

st.header("Customer Interaction Data")
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
ProductPitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{ 
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

if st.button("Predict"):

    # Make prediction
    prediction_proba = model.predict_proba(input_data)[:, 1][0] # Get probability of class 1
    
    # Classification threshold
    classification_threshold = 0.45   # Using the same threshold as in training

    # Converting probability to a binary prediction based on the threshold
    prediction = 1 if prediction_proba >= classification_threshold else 0

    result = "will purchase" if prediction == 1 else "will NOT purchase"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts the customer **{result}** the package with a probability of {prediction_proba:.2f}.")
    else:
        st.info(f"The model predicts the customer **{result}** the package with a probability of {prediction_proba:.2f}.")
