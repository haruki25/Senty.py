import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters
    text = re.sub("[^A-Za-z0-9 ]+", " ", text)
    return text


def load_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, header=None)
    data.columns = ["id", "information", "type", "text"]
    data["processed_text"] = data["text"].apply(preprocess_text)
    return data


def create_bow(data, ngram_range=(1, 4), use_stopwords=False):
    if use_stopwords:
        stop_words = stopwords.words("english")
    else:
        stop_words = None

    bow = CountVectorizer(
        tokenizer=word_tokenize, stop_words=stop_words, ngram_range=ngram_range
    )
    X = bow.fit_transform(data["processed_text"])
    y = data["type"]
    return X, y, bow


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    model = XGBClassifier(
        objective="multi:softmax",
        n_estimators=1000,
        colsample_bytree=0.6,
        subsample=0.6,
    )
    model.fit(X_train, y_train_num)
    return model, le


def evaluate_model(model, X_test, y_test, le=None):
    if le is not None:
        y_test = le.transform(y_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    return accuracy


def predict_sentiment(text, model, bow, le=None):
    processed_text = preprocess_text(text)
    X = bow.transform([processed_text])
    prediction = model.predict(X)[0]
    if le is not None:
        prediction = le.inverse_transform([prediction])[0]
    return prediction


def save_model(model, bow, data, file_path):
    joblib.dump((model, bow, data), file_path)
    print(f"Model and data saved to {file_path}")


def load_model(file_path):
    model, bow, data = joblib.load(file_path)
    print(f"Model and data loaded from {file_path}")
    return model, bow, data


def train_and_save_model(train_data_path, model_save_path):
    # Load and preprocess data
    data = load_data(train_data_path)

    # Create Bag of Words
    X, y, bow = create_bow(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression model
    lr_model = train_logistic_regression(X_train, y_train)
    lr_accuracy = evaluate_model(lr_model, X_test, y_test)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}%")

    # Save the model, bow, and data
    save_model(lr_model, bow, data, model_save_path)

    return lr_model, bow, data
