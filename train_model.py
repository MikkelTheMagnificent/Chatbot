import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

# Load preprocessed data including test data
with open('train_data.pkl', 'rb') as file:
    train_x, train_y, test_x, test_y = pickle.load(file)

# Load classes
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# Building the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # First layer
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) # Second layer
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # Output layer

# Compiling the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # Hyperparameter tuning

# Training the model with validation data
hist = model.fit(train_x, train_y, epochs=1000, batch_size=5, verbose=1, validation_data=(test_x, test_y))
model.save('chatbot_model.h5', hist)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
print(f"Test Accuracy: {test_accuracy}")

# Predict classes on test data
y_pred = model.predict(test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_y, axis=1)


unique_classes = np.unique(y_true)

# Generate a classification report with the correct labels
report = classification_report(y_true, y_pred_classes, labels=unique_classes, target_names=[classes[i] for i in unique_classes])
print(report)

print("Training complete and performance metrics calculated.")
