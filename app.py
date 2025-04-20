import numpy as np
import pandas as pd
from flask import Flask, render_template
import requests
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from sklearn.metrics import mean_absolute_error
import time
from requests.exceptions import HTTPError
import pickle
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CACHE_DIR = 'cache'
COINS_CACHE_FILE = os.path.join(CACHE_DIR, 'coins_cache.pkl')
HISTORICAL_CACHE_FILE = os.path.join(CACHE_DIR, 'historical_{coin_id}_{days}.pkl')
CACHE_TIMEOUT = 300  # 5 minutes

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def load_cached_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cache = pickle.load(f)
            if (datetime.now() - cache['timestamp']).total_seconds() < CACHE_TIMEOUT:
                logger.debug(f"Loaded data from cache: {filename}")
                return cache['data']
    return None

def save_cached_data(filename, data):
    cache = {'data': data, 'timestamp': datetime.now()}
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)
    logger.debug(f"Saved data to cache: {filename}")

def fetch_coins():
    cached_coins = load_cached_data(COINS_CACHE_FILE)
    if cached_coins:
        return cached_coins
    url = "https://api.coingecko.com/api/v3/coins/markets?order=market_cap_desc&per_page=5&page=1&vs_currency=usd"
    coins = fetch_with_retry(url)
    if coins:
        save_cached_data(COINS_CACHE_FILE, coins)
    return coins

def fetch_with_retry(url, max_retries=6, backoff_factor=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            if response.status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Rate limit hit for {url}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching {url}: {e}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return []
    logger.error(f"Max retries reached for {url}")
    return []

def fetch_historical_data(coin_id, days=30):
    cache_file = HISTORICAL_CACHE_FILE.format(coin_id=coin_id, days=days)
    cached_data = load_cached_data(cache_file)
    if cached_data:
        return cached_data
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    data = fetch_with_retry(url)
    if not data:
        return []
    prices = data.get('prices', [])
    daily_data = []
    for i in range(0, len(prices), 24):
        if i < len(prices):
            daily_data.append(prices[i][1])
    logger.debug(f"Daily data for {coin_id}: {len(daily_data)} days")
    if daily_data:
        save_cached_data(cache_file, daily_data)
    return daily_data

def prepare_data(data, time_steps=60):
    if len(data) < time_steps:
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(time_steps):
    model = Sequential([
        Input(shape=(time_steps, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(historical_data, days=7, time_steps=60):
    if len(historical_data) < time_steps + 7:
        return None
    X, y, scaler = prepare_data(historical_data[:-7], time_steps)
    if X is None:
        return None
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = build_lstm_model(time_steps)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    logger.debug(f"MAE for validation: {mae:.2f}")
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

def calculate_trends(predictions, threshold=0.003):
    trends = []
    for i in range(len(predictions)):
        if i == 0:
            trends.append('—')
        else:
            change = (predictions[i] - predictions[i-1]) / predictions[i-1]
            if change > threshold:
                trends.append('▲')
            elif change < -threshold:
                trends.append('▼')
            else:
                trends.append('—')
    return trends

def calculate_analysis(predictions):
    if len(predictions) < 2:
        return {'seven_day_change': None, 'recommendation': 'Hold'}
    seven_day_change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
    recommendation = 'Hold'
    if seven_day_change > 2:
        recommendation = 'Buy'
    elif seven_day_change < -2:
        recommendation = 'Sell'
    return {'seven_day_change': round(seven_day_change, 2), 'recommendation': recommendation}

def format_market_cap(value):
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    else:
        return f"${value / 1_000_000:.2f}M"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    coins = fetch_coins()
    if not coins:
        return "Error fetching coin data", 500
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
            continue
        if coin_id == 'tether':
            predictions = [1.0] * 7
        else:
            predictions = predict_future_prices(historical_data, days=7, time_steps=time_steps)
            if predictions is None:
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

@app.route('/prices')
def prices():
    coins_data = fetch_coins()
    if not coins_data:
        return "Error fetching coin data", 500
    coins = []
    chart_data = {'dates': [], 'prices': {}, 'rsi': {}, 'volumes': {}}
    start_date = datetime.now() - timedelta(days=7)
    chart_data['dates'] = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    total_market_cap = 0
    weighted_change_24h = 0
    for rank, coin in enumerate(coins_data, 1):
        coin_id = coin['id']
        coin_name = coin['name']
        coin_symbol = coin['symbol'].upper()
        market_cap = coin.get('market_cap', 0)
        price = coin.get('current_price', 0)
        price_change_24h = coin.get('price_change_percentage_24h', 0)
        total_market_cap += market_cap
        weighted_change_24h += coin.get('price_change_percentage_24h', 0) * market_cap
        coins.append({
            'rank': rank,
            'name': coin_name,
            'symbol': coin_symbol,
            'market_cap': market_cap,
            'market_cap_formatted': format_market_cap(market_cap),
            'current_price': price,
            'price_change_percentage_24h': price_change_24h
        })
        data = fetch_historical_data(coin_id, days=30)
        if not data or len(data) < 14:
            continue
        daily_prices = data[-7:]
        chart_data['prices'][coin_name] = [round(float(p), 2) for p in daily_prices]
        rsi_values = calculate_rsi(data, period=7)
        chart_data['rsi'][coin_name] = [round(float(r), 2) if r is not None else 0.0 for r in rsi_values[-7:]]
        chart_data['volumes'][coin_name] = fetch_volumes(coin_id, days=7)
    total_market_cap_change_24h = (weighted_change_24h / total_market_cap) if total_market_cap else 0
    logger.debug(f"Coins: {coins}")
    logger.debug(f"Chart data: {chart_data}")
    return render_template('prices.html', coins=coins, chart_data=chart_data, 
                         total_market_cap=total_market_cap, 
                         total_market_cap_change_24h=total_market_cap_change_24h,
                         total_market_cap_formatted=format_market_cap(total_market_cap))

def calculate_rsi(prices, period=7):
    if len(prices) < period + 1:
        return [None] * len(prices)
    deltas = np.diff(prices)
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rs = [None] * period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs_val = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs_val))
        rs.append(round(rsi, 2))
    return rs[-len(prices):]

def fetch_volumes(coin_id, days=7):
    cache_file = HISTORICAL_CACHE_FILE.format(coin_id=f"volume_{coin_id}", days=days)
    cached_data = load_cached_data(cache_file)
    if cached_data:
        return cached_data
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    data = fetch_with_retry(url)
    if not data:
        return [0] * 7
    volumes = [v[1] for v in data.get('total_volumes', [])]
    daily_volumes = []
    for i in range(0, len(volumes), 24):
        if i < len(volumes):
            daily_volumes.append(volumes[i])
    daily_volumes = daily_volumes[-7:] if len(daily_volumes) >= 7 else [0] * 7
    if daily_volumes:
        save_cached_data(cache_file, daily_volumes)
    return daily_volumes

def format_number(value):
    return "{:,.2f}".format(value)

app.jinja_env.filters['format_number'] = format_number
app.jinja_env.filters['format_market_cap'] = format_market_cap

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))