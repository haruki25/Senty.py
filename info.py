import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('tweets.csv', header=0, encoding='utf-8')

# Split the data into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(data['tweet'], data['label'], test_size=0.2, random_state=42)

# Preprocess the data using NLTK
stop_words = nltk.corpus.stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words, tokenizer=tokenizer.tokenize)
train_features = vectorizer.fit_transform(train_data)
val_features = vectorizer.transform(val_data)

# Train a classifier using scikit-learn
clf = MultinomialNB()
clf.fit(train_features, train_labels)

# Test the classifier on the validation set
val_pred = clf.predict(val_features)
accuracy = accuracy_score(val_labels, val_pred)
precision = precision_score(val_labels, val_pred)
recall = recall_score(val_labels, val_pred)
f1 = f1_score(val_labels, val_pred)
cm = confusion_matrix(val_labels, val_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'Confusion matrix: \n{cm}')

# Load the test data
test_data = pd.read_csv('test_tweets.csv', header=0, encoding='utf-8')

# Preprocess the test data using NLTK
test_features = vectorizer.transform(test_data['tweet'])

# Predict the labels of the test data using the trained classifier
test_pred = clf.predict(test_features)

# Save the predictions to a CSV file
test_data['label'] = test_pred
test_data.to_csv('test_predictions.csv', index=False)
