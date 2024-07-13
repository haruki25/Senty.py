import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
from nrclex import NRCLex
from streamlit_extras.chart_container import chart_container
from wordcloud import WordCloud
import io
import base64
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import os
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize

# Import functions from your model file
from utils.train_model import (
    predict_sentiment,
    load_model,
    train_and_save_model,
)

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")


def create_sentiment_gauge(polarity):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=polarity,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Sentiment Polarity"},
            gauge={
                "axis": {"range": [-1, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [-1, -0.5], "color": "red"},
                    {"range": [-0.5, 0.5], "color": "gray"},
                    {"range": [0.5, 1], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": polarity,
                },
            },
        )
    )
    fig.update_layout(height=300)
    return fig


def sentence_level_sentiment(text):
    sentences = sent_tokenize(text)
    sentiments = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        sentiment = blob.sentiment.polarity
        sentiments.append({"sentence": sentence, "sentiment": sentiment})
    return pd.DataFrame(sentiments)


@st.cache_resource
def load_or_train_model():
    train_data_path = "datasets/twitter_training.csv"
    model_save_path = "sentiment_model.joblib"

    if os.path.exists(model_save_path):
        lr_model, bow, data = load_model(model_save_path)
    else:
        lr_model, bow, data = train_and_save_model(train_data_path, model_save_path)

    return lr_model, bow, data


# Load pre-trained model and data
lr_model, bow, data = load_or_train_model()
processed_text = data["processed_text"]


@st.cache_data
def generate_word_cloud(text):
    if not text.strip():
        return None
    wordcloud = WordCloud(
        width=800, height=400, background_color=None, mode="RGBA"
    ).generate(text)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format="PNG")
    return base64.b64encode(img.getvalue()).decode()


def analyze_sentiment(text, model, bow):
    prediction = predict_sentiment(text, model, bow)
    polarity = TextBlob(text).sentiment.polarity
    emotion_analyzer = NRCLex(text)
    emotions = emotion_analyzer.affect_frequencies
    return prediction, polarity, emotions


def plot_emotion_distribution(emotions):
    emotion_df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Intensity"])
    emotion_df = emotion_df.sort_values("Intensity", ascending=False)
    with chart_container(emotion_df):
        fig = px.bar(emotion_df, x="Emotion", y="Intensity")
        st.plotly_chart(fig)


def plot_class_distribution(data):
    class_distribution = data["type"].value_counts().reset_index()
    class_distribution.columns = ["Sentiment", "Count"]
    fig = px.pie(
        class_distribution,
        values="Count",
        names="Sentiment",
    )
    return fig


def create_plutchik_wheel(emotions):
    emotion_mapping = {
        "joy": "Joy",
        "trust": "Trust",
        "fear": "Fear",
        "surprise": "Surprise",
        "sadness": "Sadness",
        "disgust": "Disgust",
        "anger": "Anger",
        "anticipation": "Anticipation",
    }

    emotions_dict = {
        emotion_mapping.get(k, k): v
        for k, v in emotions.items()
        if k in emotion_mapping
    }

    colors = [
        "#FFFF00",
        "#9DE093",
        "#00C4CC",
        "#8D5AFF",
        "#0000FF",
        "#C355FF",
        "#FF0000",
        "#FF8C00",
    ]

    fig = go.Figure(
        go.Barpolar(
            r=[emotions_dict.get(emotion, 0) for emotion in emotion_mapping.values()],
            theta=list(emotion_mapping.values()),
            width=[0.8] * 8,
            marker_color=colors,
            marker_line_color="black",
            marker_line_width=1,
            opacity=0.6,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(emotions_dict.values())]),
            angularaxis=dict(direction="clockwise"),
        ),
        showlegend=False,
    )
    emotions_dict = pd.DataFrame(
        list(emotions_dict.items()), columns=["Emotion", "Intensity"]
    )
    with chart_container(emotions_dict):
        st.plotly_chart(fig)


def display_sentiment_analysis(text):
    st.text("Text: " + text)
    prediction, polarity, emotions = analyze_sentiment(text, lr_model, bow)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Analysis")
        sentiment_color = {
            "positive": "#00FF00",
            "negative": "red",
            "neutral": "gray",
            "irrelevant": "orange",
        }
        st.markdown(
            f"Sentiment: <span style='color: {sentiment_color.get(prediction.lower(), 'white')};'>{prediction}</span>",
            unsafe_allow_html=True,
        )
        st.text(f"Polarity Score: {polarity:.2f}")
        st.progress((polarity + 1) / 2)  # Convert polarity to 0-1 range
        st.info(
            """
            **Polarity** ranges from -1 (very negative) to 1 (very positive).
            A score close to 0 indicates a neutral sentiment.
            """
        )

        # Sentiment Gauge Chart
        gauge_fig = create_sentiment_gauge(polarity)
        st.plotly_chart(gauge_fig)

        # Sentence-level Sentiment Breakdown
        st.text("Sentence-level Sentiment")
        sentence_df = sentence_level_sentiment(text)
        st.dataframe(sentence_df)

    with col2:
        st.subheader("Emotion Analysis")
        st.text("Emotion Distribution")
        plot_emotion_distribution(emotions)
        st.text("Plutchik's Wheel of Emotions")
        create_plutchik_wheel(emotions)


@st.cache_data
def get_classification_report(y, y_pred):
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df


@st.cache_data
def get_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    return cm


@st.cache_data
def get_feature_importance(_model, _feature_names):
    feature_importance = pd.DataFrame(
        {"feature": _feature_names, "importance": np.abs(_model.coef_[0])}
    )
    return feature_importance.sort_values("importance", ascending=False)
