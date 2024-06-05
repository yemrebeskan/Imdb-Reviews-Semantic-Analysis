from datasets import load_dataset
reviews = load_dataset('imdb')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, LSTM, Dropout
import math
!pip install tensorflow
import tensorflow as tf
for key, value in reviews.items():
    print(f"Length of list in '{key}': {len(value)}")

# Dataset'ten verileri sözlüklere çekme
train_reviews = reviews['train']
X_train = train_reviews['text']  # metin sütunu
y_train = train_reviews['label'];
test_reviews = reviews['test']
X_test = test_reviews['text']
y_test = test_reviews['label']
unsupervised_reviews = reviews['unsupervised']

# DataFrame oluşturma
df_train = pd.DataFrame(train_reviews)
df_test = pd.DataFrame(test_reviews)
df_unsupervised = pd.DataFrame(unsupervised_reviews)

print(df_train.head())
# Metin verilerini sayısal formata dönüştürmek için Tokenizer kullanma
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
max_words = 5000  # En sık kullanılan 5000 kelimeyi kullan
max_len = 100  # Maksimum dizi uzunluğu

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Etiketleri sayısal formata dönüştürmek için LabelEncoder kullanma
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)
# Modeli oluşturma
from sklearn.metrics import accuracy_score
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_len))  # Embedding boyutunu artır
model.add(LSTM(128))  # Daha güçlü bir LSTM katmanı
model.add(Dropout(0.5))  # Overfitting'i önlemek için Dropout
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy()])

# Modeli eğitme
model.fit(X_train_pad, y_train_enc, epochs=10, batch_size=64, validation_split=0.2)  # Batch boyutunu artır

# Tahmin yapma ve başarı oranını hesaplama
y_train_pred = (model.predict(X_train_pad) > 0.5).astype("int32")
y_test_pred = (model.predict(X_test_pad) > 0.5).astype("int32")

train_acc = accuracy_score(y_train_enc, y_train_pred)
test_acc = accuracy_score(y_test_enc, y_test_pred)

print(f'Train Accuracy: {train_acc:.2f}')
print(f'Test Accuracy: {test_acc:.2f}')


model.save('/content/drive/MyDrive/lstm model')

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Önceden eğitilmiş tokenizer'ı kullanarak metni tokenize etme ve padding yapma
def preprocess_text(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences

# Metin üzerinde duyarlılık tahmini yapma fonksiyonu
def predict_sentiment(text, tokenizer,model, max_len):
    preprocessed_text = preprocess_text(text, tokenizer, max_len)
    prediction = model.predict(preprocessed_text)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    return sentiment, float(prediction)

# Kullanıcıdan metin almak için input kullanma
input_text = input("Please enter a text to analyze sentiment: ")

# Tahmin yapma
sentiment, confidence = predict_sentiment(input_text, tokenizer, model, max_len)

# Sonucu yazdırma
print(f"The sentiment of the text is {sentiment} with a confidence of {confidence:.4f}")
