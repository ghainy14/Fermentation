{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c5e39edf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "\n",
    "# Load your saved model\n",
    "with open('Ferment_linear_model.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "493b1a77",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1504123275.py, line 29)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"C:\\Users\\User1\\AppData\\Local\\Temp\\ipykernel_1284\\1504123275.py\"\u001b[1;36m, line \u001b[1;32m29\u001b[0m\n\u001b[1;33m    streamlit run app.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "\n",
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
    "    feature1 = st.text_input('WET BIOMASS(g)', min_value=0, max_value=100)\n",
    "    feature2 = st.text_input('DRY BIOMASS(g)', min_value=0, max_value=100)\n",
    "    feature3 = st.text_input('CARBON SOURCE', min_value=0, max_value=100)\n",
    "\n",
    "    # Button to trigger prediction\n",
    "    if st.button('Predict'):\n",
    "        # Make prediction\n",
    "        input_data = [[feature1, feature2,feature3]]  # Adjust according to your model's input format\n",
    "        prediction = predict(input_data)\n",
    "\n",
    "        # Display prediction result\n",
    "        st.write('Prediction:', prediction)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n",
    "    \n",
    "streamlit run app.py\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25e86733",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7dc3aac",
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
