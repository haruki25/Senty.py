import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Senty.py",
    page_icon="assets/logo.png",
    initial_sidebar_state="expanded",
)
from utils.batch_processing import process_batch
import pandas as pd
import nltk
from utils.audio import transcribe_audio
from utils.utils import (
    display_sentiment_analysis,
    load_or_train_model,
)

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load pre-trained model and data
lr_model, bow, data = load_or_train_model()
processed_text = data["processed_text"]

# Sidebar
st.sidebar.title("Senty.py")

# Keep some basic info in the sidebar
st.sidebar.header("Quick Info")
st.sidebar.metric("Total Samples", len(data))

# About section
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses a pre-trained Logistic Regression model to predict sentiment from text input.
    It also provides emotion analysis using NRCLex.
    - **Sentiment**: Overall positive, negative, or neutral tone of the text.
    - **Polarity**: A value between -1 (very negative) and 1 (very positive).
    - **Emotions**: Intensity of different emotions in the text.
    """
)

# Main content
st.title("Senty.py: Sentiment Analysis Tool")

# Radio buttons for text and batch analysis
analysis_type = st.radio(
    "Choose analysis type:",
    ("Text Analysis", "Batch Analysis", "Audio Analysis"),
)

if analysis_type == "Text Analysis":
    user_input = st.text_area("Enter your text here:", key="user_input")
    sample_text = "I love this product! It's amazing and works perfectly."

    if st.button(
        "Analyze Sentiment",
        key="analyze_sentiment",
        disabled=user_input.strip() == "",
    ):
        display_sentiment_analysis(user_input)

    if st.button("Analyze Sample Text", key="analyze_sample_text"):
        display_sentiment_analysis(sample_text)

    # File upload for sentiment analysis
    st.markdown("---")
    st.subheader("Upload a text file for analysis")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8")
        display_sentiment_analysis(file_text)

elif analysis_type == "Batch Analysis":
    st.subheader("Batch Processing")
    uploaded_file = st.file_uploader(
        "Upload a CSV file for batch processing", type=["csv"]
    )
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        if "text" in batch_data.columns:
            processed_data = process_batch(batch_data, lr_model, bow)
            st.write("Batch Processing Results:")
            st.dataframe(processed_data)

            csv = processed_data.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="batch_sentiment_analysis_results.csv",
                mime="text/csv",
            )
        else:
            st.error("The CSV file must contain a 'text' column.")

elif analysis_type == "Audio Analysis":
    st.subheader("Audio Analysis")
    st.write("Upload an audio file for transcription and sentiment analysis.")
    audio_file = st.file_uploader("Choose an audio file", type=["wav"])
    if audio_file is not None:
        audio_text, audio_error = transcribe_audio(audio_file)
        if audio_text:
            display_sentiment_analysis(audio_text)
        else:
            st.error(
                "An error occurred during audio transcription. Error: " + audio_error
            )
