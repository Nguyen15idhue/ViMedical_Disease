import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# ==================== TRADITIONAL ML: CountVectorizer ====================
print("\n" + "="*50)
print("Creating data for Traditional ML models...")

# Vectorize with CountVectorizer (binary)
vectorizer = CountVectorizer(max_features=1000, binary=True)
X_train_ml = vectorizer.fit_transform(train_data['Processed_Question']).toarray()
X_test_ml = vectorizer.transform(test_data['Processed_Question']).toarray()

print("Traditional ML - Vocabulary size:", len(vectorizer.vocabulary_))
print("Traditional ML - X_train shape:", X_train_ml.shape)

# ==================== DEEP LEARNING: Tokenizer ====================
print("\n" + "="*50)
print("Creating data for Deep Learning models...")

# Create tokenizer for sequences
MAX_VOCAB_SIZE = 10000  # Tăng lên để có đủ từ vựng
MAX_SEQUENCE_LENGTH = 1000

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['Processed_Question'])

# Convert text to sequences
X_train_sequences = tokenizer.texts_to_sequences(train_data['Processed_Question'])
X_test_sequences = tokenizer.texts_to_sequences(test_data['Processed_Question'])

# Pad sequences to same length
X_train_dl = pad_sequences(X_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_dl = pad_sequences(X_test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

# Get vocabulary size (add 1 for padding token)
vocab_size = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)

print("Deep Learning - Vocabulary size:", vocab_size)
print("Deep Learning - Word index size:", len(tokenizer.word_index))
print("Deep Learning - X_train shape:", X_train_dl.shape)
print("Deep Learning - Max sequence length:", MAX_SEQUENCE_LENGTH)
print("Deep Learning - Sample sequences:")
print("Original:", train_data['Processed_Question'].iloc[0])
print("Sequence:", X_train_dl[0][:20])  # Show first 20 tokens

# ==================== ENCODE LABELS ====================
# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['Disease'])
y_test = label_encoder.transform(test_data['Disease'])

print("\nLabels - Number of classes:", len(np.unique(y_train)))
print("Labels - Classes:", label_encoder.classes_[:10])  # Show first 10 classes

# ==================== SAVE PROCESSED DATA ====================
print("\n" + "="*50)
print("Saving processed data...")

# Save Traditional ML data (for step3)
np.save('data/X_train.npy', X_train_ml)
np.save('data/y_train.npy', y_train)
np.save('data/X_test.npy', X_test_ml)
np.save('data/y_test.npy', y_test)

# Save Deep Learning data (for step4)
np.save('data/X_train_dl.npy', X_train_dl)
np.save('data/y_train_dl.npy', y_train)
np.save('data/X_test_dl.npy', X_test_dl)
np.save('data/y_test_dl.npy', y_test)

# Save tokenizer and metadata
import joblib
joblib.dump(tokenizer, 'data/tokenizer.joblib')
joblib.dump(label_encoder, 'data/label_encoder.joblib')
joblib.dump({
    'vocab_size': vocab_size,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'max_vocab_size': MAX_VOCAB_SIZE
}, 'data/dl_metadata.joblib')

print("✅ Traditional ML data saved to data/X_train.npy, data/y_train.npy, etc.")
print("✅ Deep Learning data saved to data/X_train_dl.npy, data/y_train_dl.npy, etc.")
print("✅ Tokenizer and metadata saved to data/")

print("\n" + "="*50)
print("SUMMARY:")
print(f"Traditional ML: {X_train_ml.shape[0]} samples, {X_train_ml.shape[1]} features")
print(f"Deep Learning: {X_train_dl.shape[0]} samples, {X_train_dl.shape[1]} sequence length, {vocab_size} vocab size")
print(f"Classes: {len(np.unique(y_train))} disease categories")