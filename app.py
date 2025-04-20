import logging
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import requests
from requests.exceptions import HTTPError
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time

# ... (other imports and code remain unchanged)

def predict_future_prices(historical_data, days=7, time_steps=60):
    if len(historical_data) < time_steps:
        logger.warning(f"Insufficient historical data for prediction: {len(historical_data)} < {time_steps}")
        return None
    try:
        model_path = 'models/lstm_bitcoin.h5'
        scaler_path = 'models/scaler.pkl'
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.error(f"Model or scaler file missing: {model_path}, {scaler_path}")
            return None
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model/scaler: {e}")
        return None
    last_sequence = scaler.transform(np.array(historical_data[-time_steps:]).reshape(-1, 1))
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        current_sequence_reshaped = current_sequence.reshape(1, time_steps, 1)
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

@app.route('/predictions')
def predictions():
    coins = fetch_coins()
    if not coins:
        logger.error("Failed to fetch coin data")
        return render_template('predictions.html', prediction_data={'coins': [], 'dates': [], 'error': "Unable to fetch coin data"}), 200
    prediction_data = {'coins': [], 'dates': []}
    start_date = datetime(2025, 4, 20)
    prediction_data['dates'] = [(start_date + timedelta(days=i)).strftime('%b %d') for i in range(7)]
    time_steps = 60
    for coin in coins:
        coin_id = coin['id']
        coin_name = coin['name']
        coin_symbol = coin['symbol'].upper()
        historical_data = fetch_historical_data(coin_id, days=90)
        if not historical_data:
            logger.warning(f"No historical data for {coin_name}")
            continue
        if coin_id == 'tether':
            predictions = [1.0] * 7
        else:
            predictions = predict_future_prices(historical_data, days=7, time_steps=time_steps)
            if predictions is None:
                logger.warning(f"Prediction failed for {coin_name}")
                continue
            predictions = [round(float(p), 2) for p in predictions]
        trends = calculate_trends(predictions, threshold=0.003)
        analysis = calculate_analysis(predictions)
        chart_data = {
            'historical': historical_data[-30:],
            'predicted': predictions,
            'historical_dates': [(datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d') for i in range(30)][::-1],
            'predicted_dates': prediction_data['dates']
        }
        prediction_data['coins'].append({
            'name': coin_name,
            'symbol': coin_symbol,
            'predictions': predictions,
            'trends': trends,
            'analysis': analysis,
            'chart_data': chart_data
        })
    logger.debug(f"Prediction data: {prediction_data}")
    return render_template('predictions.html', prediction_data=prediction_data)