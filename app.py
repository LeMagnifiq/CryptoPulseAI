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
import random
from queue import Queue
from threading import Lock

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define format_number filter
def format_number(value):
    try:
        return "{:,.2f}".format(float(value))
    except (ValueError, TypeError):
        return value
app.jinja_env.filters['format_number'] = format_number

# Format market cap (e.g., $2,781,836,034,315 -> $2.8T)
def format_market_cap(value):
    try:
        value = float(value)
        if value >= 1_000_000_000_000:  # Trillions
            return f"${value / 1_000_000_000_000:.1f}T"
        elif value >= 1_000_000_000:  # Billions
            return f"${value / 1_000_000_000:.1f}B"
        elif value >= 1_000_000:  # Millions
            return f"${value / 1_000_000:.1f}M"
        else:
            return f"${value:,.0f}"
    except (ValueError, TypeError):
        return "$0"

# Calculate RSI (Relative Strength Index)
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return [0.0] * len(prices)
    prices = np.array(prices, dtype=float)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    # Avoid division by zero for stablecoins
    rs = np.where(avg_loss > 0.0001, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 0  # RSI undefined for first period
    return rsi.tolist()

# Cache and request queue
CACHE = {}
CACHE_TIMEOUT = 600  # 10 minutes
CACHE_FILE = "cache.pkl"  # Persistent cache
REQUEST_QUEUE = Queue()
REQUEST_LOCK = Lock()
REQUEST_DELAY = 10.0  # Increased delay to avoid 429

# Load cache from file
def load_cache():
    global CACHE
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                CACHE = pickle.load(f)
            logger.debug("Loaded cache from file")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")

# Save cache to file
def save_cache():
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(CACHE, f)
        logger.debug("Saved cache to file")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# Initialize cache
load_cache()

def process_request_queue():
    while not REQUEST_QUEUE.empty():
        with REQUEST_LOCK:
            if not REQUEST_QUEUE.empty():
                url, timeout = REQUEST_QUEUE.get()
                try:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    time.sleep(REQUEST_DELAY)
                    return response
                except HTTPError as e:
                    logger.error(f"HTTP error in queued request {url}: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Error in queued request {url}: {e}")
                    raise
    return None

def fetch_coins():
    cache_key = "coins_markets"
    if cache_key in CACHE and time.time() - CACHE[cache_key]["timestamp"] < CACHE_TIMEOUT:
        return CACHE[cache_key]["data"]
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=5&page=1"
        REQUEST_QUEUE.put((url, 10))
        response = process_request_queue()
        if response:
            data = response.json()
            for coin in data:
                coin['market_cap_formatted'] = format_market_cap(coin['market_cap'])
            CACHE[cache_key] = {"data": data, "timestamp": time.time()}
            save_cache()
            return data
        return []
    except HTTPError as e:
        logger.error(f"HTTP error fetching coins: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching coins: {e}")
        return []

def fetch_market_data():
    cache_key = "global_market"
    if cache_key in CACHE and time.time() - CACHE[cache_key]["timestamp"] < CACHE_TIMEOUT:
        return CACHE[cache_key]["data"]
    try:
        url = "https://api.coingecko.com/api/v3/global"
        REQUEST_QUEUE.put((url, 10))
        response = process_request_queue()
        if response:
            data = response.json()
            total_market_cap = data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
            data['total_market_cap_formatted'] = format_market_cap(total_market_cap)
            CACHE[cache_key] = {"data": data, "timestamp": time.time()}
            save_cache()
            return data
        return {}
    except HTTPError as e:
        logger.error(f"HTTP error fetching market data: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return {}

def fetch_historical_data(coin_id, days=90, retries=5, backoff=20):
    cache_key = f"{coin_id}_historical_{days}"
    if cache_key in CACHE and time.time() - CACHE[cache_key]["timestamp"] < CACHE_TIMEOUT:
        return CACHE[cache_key]["data"]
    # Fallback for stablecoins
    if coin_id == 'tether':
        logger.debug(f"Using fallback data for {coin_id}")
        prices = [1.0] * (days // 24 + 1)
        volumes = [0.0] * (days // 24 + 1)  # Volumes unavailable
        CACHE[cache_key] = {"data": (prices, volumes), "timestamp": time.time()}
        save_cache()
        return prices, volumes
    for attempt in range(retries):
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            REQUEST_QUEUE.put((url, 10))
            response = process_request_queue()
            if response:
                prices = response.json().get('prices', [])
                volumes = response.json().get('total_volumes', [])
                if not prices or not volumes:
                    logger.warning(f"No price/volume data for {coin_id}")
                    return [], []
                daily_prices = [prices[i][1] for i in range(0, len(prices), 24) if i < len(prices)]
                daily_volumes = [volumes[i][1] for i in range(0, len(volumes), 24) if i < len(volumes)]
                if len(daily_prices) < (7 if days <= 7 else 30):
                    logger.warning(f"Incomplete data for {coin_id}: {len(daily_prices)} days")
                    return [], []
                CACHE[cache_key] = {
                    "data": (daily_prices, daily_volumes),
                    "timestamp": time.time()
                }
                save_cache()
                logger.debug(f"Fetched {len(daily_prices)} days for {coin_id}")
                return daily_prices, daily_volumes
            return [], []
        except HTTPError as e:
            if e.response and e.response.status_code == 429:
                logger.warning(f"Rate limit hit for {coin_id}, attempt {attempt + 1}/{retries}")
                if attempt < retries - 1:
                    sleep_time = backoff * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.debug(f"Backing off for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    continue
            logger.error(f"HTTP error fetching historical data for {coin_id}: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Error fetching historical data for {coin_id}: {e}")
            return [], []
    logger.error(f"Failed to fetch historical data for {coin_id} after {retries} attempts")
    return [], []

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
        return {"seven_day_change": None, "recommendation": "Hold"}
    seven_day_change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
    recommendation = "Buy" if seven_day_change > 5 else "Sell" if seven_day_change < -5 else "Hold"
    return {"seven_day_change": round(seven_day_change, 2), "recommendation": recommendation}

def predict_future_prices(historical_data, days=7, time_steps=60, coin_id=None):
    if len(historical_data) < time_steps:
        logger.warning(f"Insufficient historical data for {coin_id}: {len(historical_data)} < {time_steps}")
        return None
    # Handle stablecoins
    if coin_id in ['tether', 'usd-coin']:
        return [1.0] * days
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
        logger.error(f"Error loading model/scaler for {coin_id}: {e}")
        return None
    # Use coin-specific scaling
    coin_scaler = MinMaxScaler()
    last_sequence = coin_scaler.fit_transform(np.array(historical_data[-time_steps:]).reshape(-1, 1))
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        current_sequence_reshaped = current_sequence.reshape(1, time_steps, 1)
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    predictions = coin_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

@app.route('/')
def index():
    coins = fetch_coins()
    return render_template('index.html', coins=coins)

@app.route('/prices')
def prices():
    coins = fetch_coins()
    market_data = fetch_market_data()
    total_market_cap_change_24h = market_data.get('data', {}).get('market_cap_change_percentage_24h_usd', 0.0)
    total_market_cap_formatted = market_data.get('total_market_cap_formatted', '$0')
    # Build chart_data for prices.html
    chart_data = {
        'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)][::-1],
        'prices': {},
        'rsi': {},
        'volumes': {}
    }
    for coin in coins:
        coin_id = coin['id']
        coin_name = coin['name']
        # Try 90-day cache first
        prices, volumes = fetch_historical_data(coin_id, days=90)
        if prices and len(prices) >= 7:
            chart_data['prices'][coin_name] = [round(float(p), 2) for p in prices[-7:]]
            chart_data['volumes'][coin_name] = [round(float(v), 2) for v in volumes[-7:]]
            chart_data['rsi'][coin_name] = [round(float(r), 2) for r in calculate_rsi(prices[-21:], period=14)[-7:]]
        else:
            # Fallback to 7-day data
            prices, volumes = fetch_historical_data(coin_id, days=7)
            if not prices:
                logger.warning(f"No historical data for {coin_name} in /prices")
                chart_data['prices'][coin_name] = [round(float(coin['current_price']), 2)] * 7
                chart_data['volumes'][coin_name] = [0.0] * 7
                chart_data['rsi'][coin_name] = [0.0] * 7
            else:
                chart_data['prices'][coin_name] = [round(float(p), 2) for p in prices[-7:]]
                chart_data['volumes'][coin_name] = [round(float(v), 2) for v in volumes[-7:]]
                chart_data['rsi'][coin_name] = [round(float(r), 2) for r in calculate_rsi(prices[-7:], period=14)[-7:]]
    logger.debug(f"chart_data for /prices: {chart_data}")
    return render_template('prices.html', coins=coins, total_market_cap_change_24h=total_market_cap_change_24h,
                          total_market_cap_formatted=total_market_cap_formatted, chart_data=chart_data)

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
        prices, _ = fetch_historical_data(coin_id, days=90)
        if not prices:
            logger.warning(f"No historical data for {coin_name}")
            continue
        predictions = predict_future_prices(prices, days=7, time_steps=time_steps, coin_id=coin_id)
        if predictions is None:
            logger.warning(f"Prediction failed for {coin_name}")
            continue
        predictions = [round(float(p), 2) for p in predictions]
        trends = calculate_trends(predictions, threshold=0.003)
        analysis = calculate_analysis(predictions)
        chart_data = {
            'historical': prices[-30:],
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

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True)
else:
    import gunicorn.app.base
    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)
        def load(self):
            return self.application
    options = {
        'bind': '0.0.0.0:10000',
        'workers': 4,
        'timeout': 120,
        'loglevel': 'debug',
        'accesslog': '-',
        'errorlog': '-'
    }
    StandaloneApplication(app, options).run()