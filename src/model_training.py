import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import joblib
import os

def create_dataset(data, time_step=7):
    """Safe sequence generation"""
    X, y = [], []
    for i in range(len(data)-time_step-1):
        window = data.iloc[i:(i+time_step), :-1]
        if not window.empty:
            X.append(window.values)
            y.append(data.iloc[i+time_step, -1])
    
    if len(X) == 0:
        raise ValueError("No valid sequences generated")
        
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('./data/processed_data.csv')
    FEATURE_COLUMNS = joblib.load('./models/feature_columns.pkl')
    data = data[FEATURE_COLUMNS + ['target']]

    # Create sequences
    X, y = create_dataset(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    
    # Save
    os.makedirs('models', exist_ok=True)
    model.save('./models/traffic_model.h5')
