import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Load the CSV dataset
dataset = pd.read_csv('IMDB dataset.csv')

# Display the first few rows of the dataset to understand its structure
print(dataset.head())

# Assuming the dataset has columns 'review' (text) and 'sentiment' (positive/negative)
# Check if the 'sentiment' column is already in binary format (positive = 1, negative = 0)
dataset['sentiment'] = dataset['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Define the text (reviews) and labels (sentiments)
texts = dataset['review'].values
labels = dataset['sentiment'].values

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)  # Limit to the top 10,000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform input length
maxlen = 200  # Maximum sequence length
X = pad_sequences(sequences, maxlen=maxlen)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Save the model
model.save('model.keras')
print("model saved succesfully")

# Save the tokenizer using pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved as 'tokenizer.pkl'")

# Evaluate the model
score, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
