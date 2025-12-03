import os
import zipfile
import shutil
import gdown
import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertModel

# --- Configuration ---
MODEL_FOLDER = "imdb_distilbert_model"
ZIP_FILE = "imdb_distilbert_model.zip"
FILE_ID = "1pnL2VD5iK9Kk0_yJGhqLwAuOjJghLFOO"  # Google Drive file ID
MAX_LENGTH = 256

# --- Download and extract model if it doesn't exist ---
if not os.path.exists(MODEL_FOLDER):
    st.info("📦 Downloading model from Google Drive...")
    
    URL = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(URL, ZIP_FILE, quiet=False)
    
    tmp_folder = "tmp_model"
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(tmp_folder)
    
    # Handle nested folder in ZIP
    nested_folder = os.path.join(tmp_folder, "imdb_distilbert_model")
    if os.path.exists(nested_folder):
        if os.path.exists(MODEL_FOLDER):
            shutil.rmtree(MODEL_FOLDER)
        os.rename(nested_folder, MODEL_FOLDER)
    else:
        if os.path.exists(MODEL_FOLDER):
            shutil.rmtree(MODEL_FOLDER)
        os.rename(tmp_folder, MODEL_FOLDER)
    
    st.success("✅ Model downloaded and extracted successfully!")

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        loaded_model = tf.keras.models.load_model(
            MODEL_FOLDER,
            custom_objects={'TFDistilBertModel': TFDistilBertModel}
        )
        return tokenizer, loaded_model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop()

tokenizer, loaded_model = load_model_and_tokenizer()

# --- Prediction Function ---
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

# --- Streamlit UI ---

# Title
st.set_page_config(page_title="DistilBERT Movie Sentiment Analysis", layout="wide")
st.markdown(
    "<h2 style='text-align:center; font-size:30px; color: darkblue;'>🎬 DistilBERT IMDB Sentiment Analysis 🎬</h2>",
    unsafe_allow_html=True
)
st.markdown(
    '<p style="font-size:10px; color:gray; text-align:center;">© 2025 John Merwin. All rights reserved.</p>',
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center; font-size:16px;'>Enter a movie review below to get its sentiment prediction.</p>", unsafe_allow_html=True)
# st.markdown(
#     "<p style='text-align:center; font-size:12px; color:gray;'>Created by John Merwin</p>",
#     unsafe_allow_html=True
# )

# User input
user_input = st.text_area("✏️ Review Text", "Type your movie review here...", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "" or user_input.strip() == "Type your movie review here...":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            score = predict_sentiment(user_input)
            # st.write(f"Score: {score:.2f}")
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Prediction:")
                if score >= 0.5:
                    st.success(f"😊 Positive review")
                else:
                    st.error(f"😞 Negative review")

            with col2:
                st.subheader("Sentiment Score")
                progress_value = int(score * 100)  # Convert float [0,1] → int [0,100]
                st.progress(progress_value)
                st.write(f"{score:.2f}")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:14px;'>Score > 0.5 → Positive sentiment, Score ≤ 0.5 → Negative sentiment</p>",
                unsafe_allow_html=True
            )
           