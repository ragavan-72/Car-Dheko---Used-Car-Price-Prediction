import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\MD112\Capstone\Car Dheko - Used Car Price Prediction\Cardheko_model.xls")
df['Age_of_the_Car'] = 2024 - df['modelYear']
df = df.rename(columns={
    'ownerNo': 'owner',
    'Kms Driven': 'km',
    'Seating Capacity': 'seats'
})

# Fill missing values if necessary
df.fillna(method='ffill', inplace=True)


st.markdown("""
    <style>
        h1 {
            color: orange;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Used Car Price Prediction")

# Display the header and the logo side by side
col1, col2 = st.columns([3, 1])
with col1:
    st.header("Enter Car Details")
with col2:
    st.image("car_logo.png", width=100)

col1, col2 = st.columns(2)

with col1:
    bt = st.selectbox('Body Type (bt)', df['bt'].unique(), key='bt')
    transmission = st.selectbox('Transmission', df['transmission'].unique(), key='transmission')
    owner = st.number_input('Owner Number', min_value=1, max_value=8, key='owner')
    model = st.selectbox('Model', df['model'].unique(), key='model')
    modelyear = st.number_input('Model Year', min_value=int(df['modelYear'].min()), max_value=int(df['modelYear'].max()), key='modelYear')
    variantName = st.selectbox('Variant Name', df['variantName'].unique(), key='variantName')

with col2:
    City = st.selectbox('City', df['City'].unique(), key='City')
    insurance_validity = st.selectbox('Insurance Validity', df['Insurance Validity'].unique(), key='Insurance Validity')
    fuel_type = st.selectbox('Fuel Type', df['Fuel Type'].unique(), key='Fuel Type')
    seats = st.number_input('Seats', min_value=2, max_value=10, step=1, key='seats')
    Mileage = st.number_input('Mileage (kmpl)', min_value=5.0, max_value=50.0, step=0.1, key='Mileage')
    km = st.slider('Kilometers Driven (km)', min_value=0, max_value=500000, key='km')

# Prepare the input data
input_data = pd.DataFrame({
    'bt': [bt],
    'transmission': [transmission],
    'owner': [owner],
    'model': [model],
    'modelYear': [modelyear],
    'variantName': [variantName],
    'City': [City],
    'km': [km],
    'Insurance Validity': [insurance_validity],
    'Fuel Type': [fuel_type],
    'seats': [seats],
    'Mileage': [Mileage],
    'Age_of_the_Car': [2024 - modelyear]
})

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = label_encoders[col].fit_transform(df[col])

# Encode input data
for col in categorical_cols:
    if col in input_data.columns:
        input_data[col] = input_data[col].astype(str)
        input_data[col] = label_encoders[col].transform(input_data[col])

# Ensure input_data matches the order and features during training
input_data = input_data[df.drop(columns=['price']).columns]

# Train the model with the correct columns
X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(learning_rate=0.2, max_depth=5, n_estimators=200)
model.fit(X_train, y_train)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Price: ₹{int(prediction[0]):,}")