import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('ViMedical_Disease.csv')
print("Original data shape:", data.shape)
print(data.head())

# Clean data: Remove duplicates and NaN
data = data.drop_duplicates()
data = data.dropna()
print("After cleaning:", data.shape)

# Split into train/test with stratify
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['Disease'], random_state=42)
print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

# Save to CSV
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)
print("Data saved to data/train_data.csv and data/test_data.csv")