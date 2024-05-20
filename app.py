import streamlit as st
import pickle

# Load your saved model
with open('Ferment_linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define function to make predictions
def predict(input_data):
    # Preprocess input_data if necessary
    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
def main():
    st.title('Machine Learning Model Predictor')

    # Input features
    feature1 = st.number_input('WET BIOMASS(g)', min_value=0, max_value=100)
    feature2 = st.number_input('DRY BIOMASS(g)', min_value=0, max_value=100)
    feature3 = st.text_input('CARBON SOURCE')

    # Button to trigger prediction
    if st.button('Predict'):
        # Make prediction
        input_data = [[feature1, feature2,feature3]]  # Adjust according to your model's input format
        prediction = predict(input_data)

        # Display prediction result
        st.write('Prediction:', prediction)

if __name__ == "__main__":
    main()
