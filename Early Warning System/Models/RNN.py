import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split



data = pd.read_csv('./data.csv')
# data = data.sample(frac=0.9)

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
    Dense(3, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_rnn, y_train_cat, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test_rnn, y_test_cat)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')