import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('CDmodel.csv')
df['Age_of_the_Car'] = 2024 - df['modelYear']
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        h1 {
            color: #FF4500;
            text-align: center;
            font-size: 3em;
            font-family: 'Helvetica', sans-serif;
        }
        h3 {
            color: #000000;
            font-family: 'Helvetica', sans-serif;
        }
        .stSelectbox label, .stNumberInput label, .stSlider label {
            font-size: 1.2em;
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Used Car Price Prediction")

col1, col2 = st.columns([3, 1])
with col1:
    st.header("Enter Car Details")
with col2:
    st.image("https://your-image-url.png", width=100)

st.markdown("### Fill in the details below to get an estimated price for your car:")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    bt = st.selectbox('Body Type (bt)', df['bt'].unique(), key='bt')
    transmission = st.selectbox('Transmission', df['transmission'].unique(), key='transmission')
    ownerNo = st.number_input('Owner Number', min_value=1, max_value=8, key='owner')
    model = st.selectbox('Model', df['model'].unique(), key='model')
    modelyear = st.number_input('Model Year', min_value=int(df['modelYear'].min()), max_value=int(df['modelYear'].max()), key='modelyear')
    variantName = st.selectbox('Variant Name', df['variantName'].unique(), key='variantName')

with col2:
    insurance_validity = st.selectbox('Insurance Validity', df['Insurance Validity'].unique(), key='Insurance Validity')
    City = st.selectbox('City', df['City'].unique(), key='City')
    fuel_type = st.selectbox('Fuel Type', df['Fuel Type'].unique(), key='Fuel Type')
    km = st.slider('Kilometers Driven (km)', min_value=0, max_value=500000, key='km')
    seats = st.number_input('Seats', min_value=2, max_value=10, step=1, key='seats')
    mileage = st.number_input('Mileage (kmpl)', min_value=5.0, max_value=50.0, step=0.1, key='mileage')

# Create input data DataFrame
input_data = pd.DataFrame({
    'bt': [bt],
    'transmission': [transmission],
    'ownerNo': [ownerNo],
    'model': [model],
    'modelYear': [modelyear],
    'variantName': [variantName],
    'City': [City],
    'Insurance Validity' : [insurance_validity],
    'Fuel Type': [fuel_type],
    'Kms Driven': [km],
    'Seating Capacity': [seats],
    'Mileage': [mileage],
    'Age_of_the_Car': [2024 - modelyear]
})

df = input_data

try:
    with open('label_ecoder_rf.pkl', 'rb') as file:
        le = pickle.load(file)

    # Encode the categorical columns and handle unseen labels
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = df[col].astype(str)
        unseen_labels = set(df[col]) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.append(le.classes_, list(unseen_labels))
        df[col] = le.transform(df[col])

    with open('RandomForest_model.pkl', 'rb') as file:
        model = pickle.load(file)


except Exception as e:
    st.error(f"An error occurred: {e}")

button = st.button("Predict Price")
if button:
    # Make a prediction
    prediction = model.predict(df)
    st.success(f"Predicted Price: ₹{int(prediction[0]):,}")
