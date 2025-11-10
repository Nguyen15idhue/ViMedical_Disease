import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load train/test data
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # Remove stopwords (simple list)
    stopwords = ['tôi', 'đang', 'là', 'có', 'không', 'và', 'hoặc', 'nhưng', 'nếu', 'thì']
    tokens = [word for word in tokens if word not in stopwords and word.isalpha()]
    return ' '.join(tokens)

# Apply preprocessing
train_data['Processed_Question'] = train_data['Question'].apply(preprocess_text)
test_data['Processed_Question'] = test_data['Question'].apply(preprocess_text)

print("Sample processed text:")
print(train_data[['Question', 'Processed_Question']].head())

# Vectorize with CountVectorizer (binary)
vectorizer = CountVectorizer(max_features=1000, binary=True)
X_train = vectorizer.fit_transform(train_data['Processed_Question']).toarray()
X_test = vectorizer.transform(test_data['Processed_Question']).toarray()

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['Disease'])
y_test = label_encoder.transform(test_data['Disease'])

print("Vocabulary size:", len(vectorizer.vocabulary_))
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Save processed data
np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)
np.save('data/label_encoder.npy', label_encoder.classes_)

print("Processed data saved to data/")