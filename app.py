import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('temperature_prediction_model.h5')

# Load and preprocess the dataset
df = pd.read_csv('result.csv', encoding='latin1')

# Display the first few rows of the dataframe to check the column names
st.write("DataFrame columns:", df.columns.tolist())

# Ensure 'City' column is present
if 'city' not in df.columns:
    st.error("The dataframe must contain a 'city' column.")
    st.stop()

# Extract relevant features for prediction (update as per your notebook)
features = df.columns[1:]  # Assuming the first column is 'city' and the rest are features

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the entire dataset (excluding 'city' column)
scaler.fit(df[features])

def preprocess_input(city_name):
    # Filter data for the given city
    city_data = df[df['city'] == city_name]
    
    if city_data.empty:
        return None

    # Extract relevant features and scale them
    city_data = city_data[features]
    city_data_scaled = scaler.transform(city_data)
    
    return city_data_scaled

def main():
    st.title('Temperature Prediction App')
    
    city_name = st.text_input('Enter city name:', 'Hyderabad')
    
    if st.button('Predict'):
        city_data_scaled = preprocess_input(city_name)
        
        if city_data_scaled is not None:
            st.write(f"Input data shape: {city_data_scaled.shape[1]} (expected 383)")
            if city_data_scaled.shape[1] == 383:
                prediction = model.predict(city_data_scaled)
                st.write(f'Temperature prediction for {city_name}: {prediction[0][0]:.2f}')
            else:
                st.error(f"Input data shape mismatch. Expected 383 features, got {city_data_scaled.shape[1]}.")
        else:
            st.error(f'City {city_name} not found in the dataset.')

if __name__ == "__main__":
    main()
