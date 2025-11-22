import os
import zipfile
import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertModel
import gdown

# ----------------------------
# Configuration
# ----------------------------
MODEL_FOLDER = "imdb_distilbert_model"
ZIP_FILE = "imdb_distilbert_model.zip"
FILE_ID = "1pnL2VD5iK9Kk0_yJGhqLwAuOjJghLFOO"  # Google Drive file ID
MAX_LENGTH = 256

# ----------------------------
# Ensure model exists
# ----------------------------
def ensure_model_exists():
    if not os.path.exists(MODEL_FOLDER):
        st.info("📦 Downloading model from Google Drive (first run may take up to 30s)...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(MODEL_FOLDER)
        st.success("✅ Model downloaded and extracted successfully!")

# Run this first
ensure_model_exists()

# ----------------------------
# Load Model and Tokenizer
# ----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = tf.keras.models.load_model(
            MODEL_FOLDER,
            custom_objects={'TFDistilBertModel': TFDistilBertModel}
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop()

tokenizer, loaded_model = load_model_and_tokenizer()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_sentiment(text):
    encoded_input = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )
    prediction = loaded_model.predict([encoded_input["input_ids"], encoded_input["attention_mask"]])[0][0]
    return prediction

# ----------------------------
# Streamlit UI
# ----------------------------
# Header
st.markdown(
    "<h2 style='text-align:center; font-size:30px; color: darkblue;'>🎬 DistilBERT IMDB Sentiment Analysis 🎬</h2>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center; font-size:16px;'>Enter a movie review below to get its sentiment prediction.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:12px; color:gray;'>Created by John Merwin</p>", unsafe_allow_html=True)

# User input
user_input = st.text_area("✏️ Review Text", "Type your movie review here...", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "" or user_input.strip() == "Type your movie review here...":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            score = predict_sentiment(user_input)

            # Display prediction and score
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Prediction:")
                if score >= 0.5:
                    st.success(f"😊 Positive review")
                else:
                    st.error(f"😞 Negative review")

            with col2:
                st.subheader("Sentiment Score")
                progress_value = int(score * 100)  # float [0,1] -> int [0,100]
                st.progress(progress_value)
                st.write(f"{score:.2f}")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:14px;'>Score > 0.5 → Positive sentiment, Score ≤ 0.5 → Negative sentiment</p>",
                unsafe_allow_html=True
            )
