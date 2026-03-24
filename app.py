import streamlit as st
import fasttext
import joblib
import numpy as np
import re

# Set up the dashboard page
st.set_page_config(page_title="Hiligaynon Smishing Detector", page_icon="🛡️")
st.title("🛡️ Hiligaynon Smishing Detector")
st.write("Enter a text message below to check if it is Legitimate or Smishing.")

# Load the trained models (cached so they only load once)
@st.cache_resource
def load_models():
    ft = fasttext.load_model("hiligaynon_fasttext.bin")
    ensemble = joblib.load("stacking_ensemble_model.pkl")
    return ft, ensemble

try:
    ft_model, stacking_model = load_models()
except Exception as e:
    st.error("Error loading models. Ensure 'hiligaynon_fasttext.bin' and 'stacking_ensemble_model.pkl' are in the same folder as this script.")
    st.stop()

# Preprocessing function matching Chapter III methodology
def preprocess_text(text):
    # Lowercasing
    text = str(text).lower()
    # Remove URLs, special characters, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# User Interface
user_input = st.text_area("SMS Message:", placeholder="e.g. Alert Amigo! Ang imo BDO account gina-lock...")

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # 1. Clean the text
        cleaned_text = preprocess_text(user_input)
        
        # 2. Extract 300-dimensional FastText vector
        vector = ft_model.get_sentence_vector(cleaned_text)
        vector_reshaped = np.array(vector).reshape(1, -1)
        
        # 3. Predict using the Stacking Ensemble
        prediction = stacking_model.predict(vector_reshaped)[0]
        probabilities = stacking_model.predict_proba(vector_reshaped)[0]
        
        st.divider()
        
        # 4. Display the Results
        if prediction == 1:
            st.error(f"🚨 **WARNING: This is a Smishing Message!** (Confidence: {probabilities[1]*100:.2f}%)")
        else:
            st.success(f"✅ **SAFE: This is a Legitimate Message.** (Confidence: {probabilities[0]*100:.2f}%)")
            
        # Optional: Show behind-the-scenes data for the panelists
        with st.expander("View Algorithm Details"):
            st.write(f"**Cleaned Text:** {cleaned_text}")
            st.write(f"**Smishing Probability:** {probabilities[1]:.4f}")
            st.write(f"**Legitimate Probability:** {probabilities[0]:.4f}")