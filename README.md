# Hate Speech Detection in Tweets
This project is a classification task that aims to detect hate speech in tweets. The objective is to classify racist tweets from non-racist tweets based on their content. The project consists of the following steps:

Steps  | Description
------------- | -------------
Data Preprocessing  | Cleaning and transforming the raw data into a format that can be used for training the machine learning model.
Feature extraction  | Creating a numerical representation of the tweets using a bag-of-words model.
Training a machine learning model | Using the extracted features to train a classifier that can distinguish between racist and non-racist tweets.
Evaluating the model | Measuring the accuracy, precision, recall, and F1-score of the trained model on a validation set.
Testing the model | Applying the trained model to predict the labels of a test set.

## Dataset
The dataset used for this project is contained in the file tweets.csv. This dataset consists of tweets and their corresponding labels, where a label of 1 denotes a tweet that contains hate speech and a label of 0 denotes a non-hate-speech tweet. The dataset was split into a training set and a test set, with the test set containing 20% of the original data.

## Dependencies
The following Python libraries are required to run this project:
```python
pandas
nltk
scikit-learn
```
You can install these libraries using pip:
```bash
pip install pandas nltk scikit-learn
```
## Steps
Step 1: Data Preprocessing
The first step in this project is to preprocess the raw data. The preprocessing steps include:
Removing stop words and tokenizing the tweets using NLTK.
```python
stop_words = nltk.corpus.stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
```
This cell takes a raw tweet as input and returns a cleaned and tokenized version of the tweet.

Step 2: Feature Extraction
The next step is to create a numerical representation of the tweets using a bag-of-words model. This model counts the frequency of each word in the tweets and creates a sparse matrix of the counts. The CountVectorizer class from scikit-learn is used to create the bag-of-words representation of the tweets.
```python
vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words, tokenizer=tokenizer.tokenize)
train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)
```
This code takes a list of preprocessed tweets and returns a sparse matrix of the bag-of-words representation of the tweets.

Step 3: Training a Machine Learning Model
The third step is to train a machine learning model on the extracted features. In this project, we use the Naive Bayes classifier from scikit-learn to classify the tweets. The classifier is trained on the training set using the bag-of-words representation of the tweets.
```python
clf = MultinomialNB()
clf.fit(train_features, train_labels)
```
This code takes the bag-of-words representation of the training tweets and their corresponding labels as input, and returns the trained classifier.

Step 4: Evaluating the Model
The fourth step is to evaluate the performance of the trained model on a validation set. The evaluation metrics used in this project are accuracy, precision, recall, and F1-score. These metrics are calculated using scikit-learn's metrics module.
```python
val_pred = clf.predict(val_features)
accuracy = accuracy_score(val_labels, val_pred)
precision = precision_score(val_labels, val_pred)
recall = recall_score(val_labels, val_pred)
f1 = f1_score(val_labels, val_pred)
cm = confusion_matrix(val_labels, val_pred)
```
This code takes the trained classifier, the bag-of-words representation of the validation tweets, and their corresponding labels as input, and returns the calculated evaluation metrics.

## Added Features
1. Polarity Analyser
```python
def add_polarity_to_df(df):
    polarity_list = []
    for x in range(0, df.shape[0]):
        QuantTextBlob = TextBlob(df.iloc[x][0])
        measures = QuantTextBlob.sentiment.polarity
        polarity_list.append(measures)
    df['Text Polarity'] = polarity_list
    return df
polars= add_polarity_to_df(data)
```
This code defines a function ```python add_polarity_to_df() ``` that adds text polarity values to a DataFrame. The function takes a DataFrame as input and returns the modified DataFrame with an additional column named 'Text Polarity' that contains the polarity values for each row.
The function iterates over each row of the input DataFrame, creates a TextBlob object from the text in the 'tweet' column, and gets the polarity measure using the sentiment.polarity attribute of the TextBlob object. It then appends the polarity measure to a list, and finally adds the list as a new column to the input DataFrame.

2. Audio Transcribed Text Analyser
```python
r = sr.Recognizer()

# open the audio file using AudioFile context manager
with sr.AudioFile('sound.wav') as source:
    # record audio from source file
    audio = r.record(source)

# transcribe the audio using Google Speech Recognition API
try:
    text = r.recognize_google(audio)
    print("Google Speech Recognition thinks you said: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# create a new CSV file for writing
with open('output_file.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # write the header row name
    writer.writerow(['text'])
    
    # split the text into words
    words = text.split()
    while words:
        # join 10 words into a single row & make multiple rows for better application of model
        row = ' '.join(words[:10])
        writer.writerow([row])
        words = words[10:]
```
This code uses the speech_recognition library to transcribe an audio file into text using Google's Speech Recognition API. It then writes the transcribed text to a new CSV file called output_file.csv, with each row containing a maximum of 10 words.
## License

[MIT](https://github.com/haruki25/Sentiment_Analysis/blob/main/LICENSE)
