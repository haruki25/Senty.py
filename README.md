# Senty.py: Sentiment Analysis Tool

Senty.py is a comprehensive sentiment analysis application built with Python, leveraging Streamlit for an interactive web interface. It uses a pre-trained Logistic Regression model for sentiment prediction, NRCLex for emotion analysis, and provides functionalities for text, batch, and audio analysis.

## Features

- **Text Analysis**: Analyze the sentiment of a single text input.
- **Batch Analysis**: Process and analyze sentiment for multiple texts in a CSV file.
- **Audio Analysis**: Transcribe and analyze sentiment from an audio file.
- **Sentiment Gauge**: Visual representation of sentiment polarity.
- **Emotion Distribution**: Charts showing the intensity of different emotions in the text.
- **Word Clouds**: Generate word clouds for different sentiment categories.
- **Model Information**: Insights into the model's performance, including a confusion matrix and class distribution.

## Installation

To run Senty.py locally, you need Python 3.6 or later. Clone the repository and install the required dependencies.

```bash
git clone https://github.com/haruki25/Sentiment_Analysis.git
cd Sentiment_Analysis
pip install -r requirements.txt
```

### Dependencies

- Streamlit
- Pandas
- NLTK
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn
- WordCloud
- NRCLex

## Usage

After installing the dependencies, you can start the Streamlit application by running:

```bash
streamlit run App.py
```

Navigate to `http://localhost:8501` in your web browser to interact with the application.

## Structure

- `App.py`: The main Streamlit application file. (Homepage)
- `utils/`: Contains utility scripts for sentiment analysis, batch processing, and model training.
  - `batch_processing.py`: Functions for processing batches of text for sentiment analysis.
  - `train_model.py`: Scripts for training the sentiment analysis model.
  - `utils.py`: Various utility functions including sentiment gauge creation, word cloud generation, and more.
- `model_info.py`: Streamlit application for displaying model information and performance metrics.
- `datasets/`: Directory for storing training datasets (included in the repository).

## Contributing

Contributions to Senty.py are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- NLTK for natural language processing tools.
- Scikit-learn for machine learning algorithms.
- Streamlit for creating an interactive web application.

Thank you for exploring Senty.py. Happy sentiment analyzing!
