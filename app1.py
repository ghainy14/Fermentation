{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b545a71b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-05-27 11:22:51.344 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\User1\\anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-05-27 11:22:51.354 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import numpy as np\n",
    "import joblib\n",
    "model = joblib.load('Ferment_DecisionTree_model.pkl')\n",
    "# Assuming the model is already loaded\n",
    "# Replace this with your model loading code\n",
    "# model = ...\n",
    "\n",
    "# Define function to make predictions\n",
    "def predict(input_data):\n",
    "    # Preprocess input_data if necessary\n",
    "    # Make predictions using the loaded model\n",
    "    prediction = model.predict(input_data)\n",
    "    return prediction\n",
    "\n",
    "# Streamlit UI\n",
    "def main():\n",
    "    st.title('Machine Learning Model Predictor')\n",
    "\n",
    "    # Input features\n",
    "    feature1 = st.number_input('WET BIOMASS(g)', min_value=0, max_value=100, step=1)\n",
    "    feature2 = st.number_input('DRY BIOMASS(g)', min_value=0, max_value=100, step=1)\n",
    "\n",
    "    # Dropdown for CARBON SOURCE\n",
    "    carbon_sources = ['Shea butter kernel extract', 'Ipomoea Batatas Peel ectract', 'Palm Fruit empty fibre']  # Example options\n",
    "    feature3 = st.selectbox('CARBON SOURCE', options=carbon_sources)\n",
    "\n",
    "    # Initialize LabelEncoder\n",
    "    label_encoder = LabelEncoder()\n",
    "    label_encoder.fit(carbon_sources)\n",
    "\n",
    "    # Button to trigger prediction\n",
    "    if st.button('Predict'):\n",
    "        # Encode the carbon source\n",
    "        encoded_feature3 = label_encoder.transform([feature3])[0]\n",
    "\n",
    "        # Make prediction\n",
    "        input_data = np.array([[feature1, feature2, encoded_feature3]])  # Adjust according to your model's input format\n",
    "        prediction = predict(input_data)\n",
    "\n",
    "        # Round prediction to the nearest whole number\n",
    "        rounded_prediction = round(prediction[0])\n",
    "\n",
    "        # Display prediction result\n",
    "        st.write('Prediction:', rounded_prediction)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0679e78c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
