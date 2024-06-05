import tensorflow as tf
import transformers
from transformers import TFBertTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
dataset = load_dataset('stanfordnlp/imdb')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

'''tokenized_train_dataset = dataset['train'].map(tokenize_function, batched=True)'''
tokenized_val_dataset = dataset['test'].select(range(5000)).map(tokenize_function, batched=True)
val_dataset = tokenized_val_dataset.to_tf_dataset(
    columns=['attention_mask', 'input_ids'],
    label_cols=['label'],
    shuffle=False,
    batch_size=8,
)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.metrics.BinaryAccuracy()]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(train_dataset, validation_data=val_dataset, epochs=1)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def evaluate_sentences(sentences):

    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='tf')


    outputs = reloaded_model(inputs)
    logits = outputs['logits']
    predictions = tf.nn.sigmoid(logits)
    return predictions


sentences = ["The movie was fantastic!",
             "I did not like the film.",
             "Overly melodramatic and predictable. The story dragged on, and the characters lacked depth. I couldn't connect with the film emotionally.",
             'The movie was meh.',
             'The movie was not bad.',
              'The movie was meh.',
              'The movie was okito.',
              'The movie was terrible...',]
predictions = evaluate_sentences(sentences)
print(predictions)