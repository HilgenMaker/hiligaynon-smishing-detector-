import streamlit as st
import fasttext
import joblib
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize

# 1. Download necessary NLTK data for tokenization
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

# 2. Setup the Page
st.set_page_config(page_title="Hiligaynon Smishing Detector", page_icon="🛡️")
st.title("🛡️ Hiligaynon Smishing Detector")
st.write("Enter a text message below to check if it is Legitimate or Smishing.")

# 3. Load the Trained Algorithm Models
@st.cache_resource
def load_models():
    # Loads the 300-dimensional dense semantic vectors
    ft = fasttext.load_model("hiligaynon_fasttext.bin")
    # Loads the Meta-Classifier (Logistic Regression combining GNB, LR, RF)
    ensemble = joblib.load("stacking_ensemble_model.pkl")
    return ft, ensemble

try:
    ft_model, stacking_model = load_models()
except Exception as e:
    st.error("Error loading models. Ensure your .bin and .pkl files are uploaded correctly.")
    st.stop()

# 4. The NLP Preprocessing Algorithm
# Defined stop words to eliminate (must match the training script)
custom_stopwords = set([
    'ang', 'mga', 'nga', 'sa', 'sang', 'kay', 'kag', 'ko', 'mo', 'ni', 
    'is', 'the', 'and', 'to', 'a', 'of', 'in', 'it', 'for', 'on'
])

def preprocess_text(text):
    # a. Lowercasing
    text = str(text).lower()
    
    # b. Special character & URL handling (Keep the domain text for analysis)
    text = re.sub(r'https?://', '', text)
    text = re.sub(r'www\.', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # c. Tokenisation using NLTK
    tokens = word_tokenize(text)
    
    # d. Stop word elimination
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]
    
    # e. Join back into preprocessed text
    return ' '.join(filtered_tokens)

# 5. User Interface & Processing
user_input = st.text_area("Raw Text Input:", placeholder="e.g. Urgent! Naka-receive ka sang P50,000 cash. Confirm diri: bit.ly/yyn6bb")

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Step 1: Preprocess the Raw Text
        cleaned_text = preprocess_text(user_input)
        
        # Step 2: Convert to 300-dimension semantic vector via FastText
        vector = ft_model.get_sentence_vector(cleaned_text)
        vector_reshaped = np.array(vector).reshape(1, -1)
        
        # Step 3: Stacking Ensemble Binary Classification
        prediction = stacking_model.predict(vector_reshaped)[0]
        probabilities = stacking_model.predict_proba(vector_reshaped)[0]
        
        st.divider()
        
        # Output: Legitimate or Smishing
        if prediction == 1:
            st.error(f"🚨 **WARNING: This is a Smishing Message!** (Confidence: {probabilities[1]*100:.2f}%)")
        else:
            st.success(f"✅ **SAFE: This is a Legitimate Message.** (Confidence: {probabilities[0]*100:.2f}%)")
            
        with st.expander("View NLP Algorithm Details"):
            st.write(f"**Original Raw Text:** {user_input}")
            st.write(f"**Preprocessed Text (Tokens & No Stop Words):** {cleaned_text}")
            st.write(f"**Smishing Probability:** {probabilities[1]:.4f}")
            st.write(f"**Legitimate Probability:** {probabilities[0]:.4f}")