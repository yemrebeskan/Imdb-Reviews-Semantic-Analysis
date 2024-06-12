# Imdb Reviews Semantic Analysis
IMDB Reviews Semantic Analysis 
This study aims to classify movie reviews from the IMDB dataset into 'positive' or 'negative' sentiments using two machine learning models: BERT and LSTM. We will evaluate which model more effectively determines sentiments expressed in textual data.
- Task: Text-classification of movie reviews
- Dataset: stanfordnlp/imdb

- BERT:
 -- Libraries: TensorFlow, Transformers
 - Pre-Trained Model: Bert Base Uncased
 - Optimizer: Adam
 - Loss Function: Binary Cross Entropy
 - Metrics: Binary Accuracy
 - Epochs: 10
 - Batch Size: 8
 - Tokenizer: Transformers Bert Tokenizer
- LSTM:
 - Architecture: Embedding, LSTM layer, Dropout, dense layer.
 - Optimizer: Adam
 - Loss Function: Binary Cross Entropy
 - Metrics: Binary Accuracy
 - Epochs: 10
 - Batch Size: 32
 - Tokenizer: Tokenizer from keras


