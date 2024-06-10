import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Load preprocessed data
with open('train_data.pkl', 'rb') as file:
    train_x, train_y = pickle.load(file)

# Building the model
model = Sequential() #The model will have layers added sequentially
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # Input layer with 128 neurons
model.add(Dropout(0.5)) # Dropout layer
model.add(Dense(64, activation='relu')) # Hidden layer with 64 neurons
model.add(Dropout(0.5)) 
model.add(Dense(len(train_y[0]), activation='softmax')) # Output layer

# Compiling the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training the model
hist = model.fit(train_x, train_y, epochs=1000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Training complete.")
