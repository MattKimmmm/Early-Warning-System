import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def rnn_result():
    #data = pd.read_csv('./data.csv')
    data = pd.read_csv('./data_balanced.csv')
    data = data.sample(frac=0.05)

    X = data.drop(['SUBJECT_ID','TARGET'], axis=1)
    y = data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train_rnn = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_rnn = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential([
        SimpleRNN(50, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), return_sequences=True),
        SimpleRNN(50),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_rnn, y_train_cat, epochs=10, validation_split=0.2)
    test_loss, test_acc = model.evaluate(X_test_rnn, y_test_cat)

    y_pred = model.predict(X_test_rnn)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    return model, y_true, y_pred_classes, test_acc, test_loss


