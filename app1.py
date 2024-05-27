import streamlit as st
import numpy as np
import joblib
import joblib

# Save the model
joblib.dump(model, 'Ferment_DecisionTree_model.pkl')

# Load the model
model = joblib.load('Ferment_DecisionTree_model.pkl')

# Define function to make predictions
def predict(input_data):
    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
def main():
    st.title('Machine Learning Model Predictor')

    # Input features
    feature1 = st.number_input('WET BIOMASS(g)', min_value=0.0, max_value=100.0, step=0.1)
    feature2 = st.number_input('DRY BIOMASS(g)', min_value=0.0, max_value=100.0, step=0.1)

    # Dropdown for CARBON SOURCE
    carbon_sources = ['Shea butter kernel extract', 'Ipomoea Batatas Peel extract', 'Palm Fruit empty fibre']  # Example options
    feature3 = st.selectbox('CARBON SOURCE', options=carbon_sources)

    # Button to trigger prediction
    if st.button('Predict'):
        # Encode the carbon source
        encoded_feature3 = carbon_sources.index(feature3)

        # Make prediction
        input_data = np.array([[feature1, feature2, encoded_feature3]])  # Adjust according to your model's input format

        # Ensure the input data shape is (n_samples, n_features)
        input_data = input_data.reshape(1, -1)  # Reshape to (1, 3) assuming 3 features

        # Make prediction
        prediction = predict(input_data)

        # Display prediction result
        st.write('Predicted day:', prediction[0])

if __name__ == "__main__":
    main()
