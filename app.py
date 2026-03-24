import streamlit as st
import fasttext
import joblib
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize

# --- ALGORITHM SETUP ---

# 1. Initialize NLP Resources (Required for the Tokenization step)
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

setup_nltk()

# 2. Define Custom Stop Words (Matches Chapter III Methodology)
# These words are eliminated as they do not contribute to smishing patterns[cite: 44, 128].
custom_stopwords = set([
    'ang', 'mga', 'nga', 'sa', 'sang', 'kay', 'kag', 'ko', 'mo', 'ni', 
    'is', 'the', 'and', 'to', 'a', 'of', 'in', 'it', 'for', 'on'
])

# 3. Load Trained Models
@st.cache_resource
def load_models():
    # FastText handles morphologically rich Hiligaynon subwords[cite: 28, 52].
    ft = fasttext.load_model("hiligaynon_fasttext.bin")
    # Stacking Ensemble combines GNB, LR, and RF[cite: 35, 135].
    ensemble = joblib.load("stacking_ensemble_model.pkl")
    return ft, ensemble

# --- THE CORE ALGORITHM (PREPROCESSING) ---

def preprocess_text(text):
    """
    Step-by-step pipeline: Lowercasing, Noise Removal, Tokenization, 
    and Stop Word Elimination[cite: 44, 124, 125].
    """
    # a. Consistency: Convert to lowercase [cite: 126]
    text = str(text).lower()
    
    # b. Noise Removal: Strip http/https but keep domain keywords like 'bit.ly' [cite: 127, 173]
    text = re.sub(r'https?://', '', text)
    text = re.sub(r'www\.', '', text)
    
    # c. Special Character & Numerical Handling [cite: 127]
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # d. Tokenization: Break text into individual units [cite: 130]
    tokens = word_tokenize(text)
    
    # e. Stop Word Elimination: Remove high-frequency non-contributing words [cite: 128]
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]
    
    # f. Final Output String [cite: 46]
    return ' '.join(filtered_tokens)

# --- USER INTERFACE ---

st.set_page_config(page_title="Hiligaynon Smishing Detector", page_icon="🛡️")
st.title("🛡️ Hiligaynon Smishing Detector")
st.subheader("BSCS Thesis Project: FastText & Stacking Ensemble")

try:
    ft_model, stacking_model = load_models()
except Exception as e:
    st.error("Model files not found. Please ensure .bin and .pkl files are in the repository.")
    st.stop()

user_input = st.text_area("Input Raw SMS Text:", placeholder="Paste the Hiligaynon or English message here...")

if st.button("Run Detection Algorithm"):
    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        # 1. Processing: Apply NLP Pipeline
        preprocessed_text = preprocess_text(user_input)
        
        # 2. Vectorization: Convert to 300D semantic vector [cite: 53, 132]
        vector = ft_model.get_sentence_vector(preprocessed_text)
        vector_reshaped = np.array(vector).reshape(1, -1)
        
        # 3. Classification: Stacking Ensemble Prediction [cite: 142, 144]
        prediction = stacking_model.predict(vector_reshaped)[0]
        probabilities = stacking_model.predict_proba(vector_reshaped)[0]
        
        st.divider()
        
        # 4. Final Output [cite: 145]
        if prediction == 1:
            st.error(f"🚨 **SMISHING DETECTED** (Confidence: {probabilities[1]*100:.2f}%)")
            st.info("This message matches known fraudulent patterns or malicious link structures.")
        else:
            st.success(f"✅ **LEGITIMATE MESSAGE** (Confidence: {probabilities[0]*100:.2f}%)")
            
        # Behind the scenes for the defense panel
        with st.expander("Technical Algorithm Breakdown"):
            st.write(f"**Preprocessed (Tokens):** {preprocessed_text}")
            st.write(f"**GNB/LR/RF Ensemble Probability:** {probabilities[1]:.4f}")
            st.write("**Model Architecture:** FastText Subword Embeddings + Stacking Meta-Classifier")
