import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
import requests
import os
import pickle  # Added import

def fetch_historical_data(coin_id='bitcoin', days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        prices = response.json().get('prices', [])
        daily_data = [prices[i][1] for i in range(0, len(prices), 24) if i < len(prices)]
        return daily_data
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def prepare_data(data, time_steps=60):
    if len(data) < time_steps:
        print(f"Insufficient data: {len(data)} < {time_steps}")
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps])
    return np.array(X), np.array(y), scaler

def build_lstm_model(time_steps):
    model = Sequential([
        Input(shape=(time_steps, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save model
historical_data = fetch_historical_data()
if not historical_data:
    print("Failed to fetch historical data.")
    exit(1)
X, y, scaler = prepare_data(historical_data[:-7])
if X is not None:
    model = build_lstm_model(time_steps=60)
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_bitcoin.h5')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved to models/")
else:
    print("Insufficient data for training.")
    exit(1)