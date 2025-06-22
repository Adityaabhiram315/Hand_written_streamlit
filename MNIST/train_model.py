import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt

def train_model_with_history():
    """Train the MNIST model and save the training history"""
    
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print("Creating model...")
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=64, 
        validation_split=0.1,
        verbose=1
    )
    
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    print("Saving model and history...")
    model.save('mnist_cnn_model.h5')
    
    # Save the history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    print("Training completed and files saved!")
    return model, history

if __name__ == "__main__":
    model, history = train_model_with_history()
