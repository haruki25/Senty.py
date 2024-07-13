from matplotlib import pyplot as plt
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Senty.py: Model Info",
    page_icon="assets/logo.png",
    initial_sidebar_state="expanded",
)
import seaborn as sns
from utils.train_model import (
    predict_sentiment,
    create_bow,
)
from utils.utils import (
    generate_word_cloud,
    get_classification_report,
    get_confusion_matrix,
    get_feature_importance,
    load_or_train_model,
    plot_class_distribution,
)

# Load pre-trained model and data
lr_model, bow, data = load_or_train_model()
processed_text = data["processed_text"]


st.title("Model Information")

# Model performance
X, y, feature_names = create_bow(data)
y_pred = lr_model.predict(X)
report = get_classification_report(y, y_pred)
cm = get_confusion_matrix(y, y_pred)

st.header("Model Performance")
st.dataframe(report, use_container_width=True)

# Confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Training data info
st.header("Training Data Info")
st.metric("Total Samples", len(data))

# Class distribution chart
class_dist_fig = plot_class_distribution(data)
st.subheader("Class Distribution")
st.plotly_chart(class_dist_fig, use_container_width=True)

# Word clouds
st.header("Word Clouds")
sentiment_categories = ["Positive", "Negative", "Irrelevant", "Neutral"]
all_text = " ".join(processed_text)

for category in sentiment_categories + ["all"]:
    if category == "all":
        category_text = all_text
    else:
        category_text = " ".join(data[data["type"] == category]["processed_text"])
    word_cloud_img = generate_word_cloud(category_text)

    with st.expander(f"{category.capitalize()} Word Cloud"):
        if word_cloud_img:
            st.image(f"data:image/png;base64,{word_cloud_img}")
        else:
            st.write(f"No data available for {category} word cloud")

    # # Feature Importance
    # st.header("Feature Importance")
    # feature_importance = get_feature_importance(lr_model, feature_names)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # feature_importance.head(20).plot(x="feature", y="importance", kind="barh", ax=ax)
    # plt.title("Top 20 Most Important Features")
    # plt.xlabel("Importance")
    # st.pyplot(fig)

# Misclassified samples
st.header("Misclassified Samples")
misclassified = data[data["type"] != y_pred]
if len(misclassified) > 0:
    sample_size = min(5, len(misclassified))
    misclassified_sample = misclassified.sample(sample_size)
    for _, row in misclassified_sample.iterrows():
        st.write(f"Text: {row['text']}")
        st.write(f"True Sentiment: {row['type']}")
        st.write(
            f"Predicted Sentiment: {predict_sentiment(row['text'], lr_model, bow)}"
        )
        st.write("---")
else:
    st.write("No misclassified samples found.")
