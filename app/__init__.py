from flask import Flask, render_template
from pycoingecko import CoinGeckoAPI
import pandas as pd
import joblib
import numpy as np
import time
import pickle
import os

def create_app():
    app = Flask(__name__, template_folder='templates')
    client = CoinGeckoAPI()
    cache_file = 'data/cache.pkl'
    cache_duration = 300  # 5 minutes

    def load_cache():
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                if time.time() - cache.get('timestamp', 0) < cache_duration:
                    return cache.get('data', {})
        return {}

    def save_cache(data):
        with open(cache_file, 'wb') as f:
            pickle.dump({'timestamp': time.time(), 'data': data}, f)

    @app.route('/')
    def home():
        return 'Hello, CryptoPulseAI!'

    @app.route('/prices')
    def prices():
        coins = ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin']
        try:
            cache = load_cache()
            cache_key = 'prices'
            if cache_key in cache:
                formatted_data = cache[cache_key]
            else:
                time.sleep(1)  # Avoid rate limit
                market_data = client.get_coins_markets(
                    vs_currency='usd',
                    ids=coins,
                    order='market_cap_desc',
                    per_page=100,
                    page=1,
                    sparkline=False,
                    price_change_percentage='24h'
                )
                formatted_data = []
                coin_map = {
                    'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL',
                    'cardano': 'ADA', 'dogecoin': 'DOGE'
                }
                for coin in coins:
                    coin_info = next((item for item in market_data if item['id'] == coin), None)
                    if coin_info:
                        formatted_data.append({
                            'name': coin.capitalize(),
                            'symbol': coin_map[coin],
                            'price': coin_info['current_price'],
                            'change_24h': round(coin_info['price_change_percentage_24h'], 2) if coin_info['price_change_percentage_24h'] is not None else 'N/A'
                        })
                    else:
                        formatted_data.append({
                            'name': coin.capitalize(),
                            'symbol': coin_map[coin],
                            'price': 'N/A',
                            'change_24h': 'N/A'
                        })
                cache[cache_key] = formatted_data
                save_cache(cache)
            return render_template('prices.html', prices=formatted_data)
        except Exception as e:
            print(f"Error in prices route: {str(e)}")
            return f"Error in prices route: {str(e)}", 500

    @app.route('/predictions')
    def predictions():
        try:
            coins = {'bitcoin': 'BTC', 'ethereum': 'ETH', 'solana': 'SOL'}
            predictions = {}
            cache = load_cache()
            for coin in coins:
                rf_model = joblib.load(f'data/models/{coin}_rf.pkl')
                xgb_model = joblib.load(f'data/models/{coin}_xgb.pkl')
                cache_key = f'{coin}_market_data'
                if cache_key in cache:
                    data = cache[cache_key]
                else:
                    time.sleep(1)  # Avoid rate limit
                    data = client.get_coin_market_chart_by_id(
                        id=coin, vs_currency='usd', days=50, interval='daily'
                    )
                    cache[cache_key] = data
                    save_cache(cache)
                df = pd.DataFrame({
                    'price': [x[1] for x in data['prices']]
                })
                df['sma10'] = df['price'].rolling(window=10).mean()
                df['sma50'] = df['price'].rolling(window=50).mean()
                delta = df['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                ema12 = df['price'].ewm(span=12, adjust=False).mean()
                ema26 = df['price'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
                df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df = df.dropna()
                features = ['sma10', 'sma50', 'rsi', 'macd', 'signal']
                X = df[features].iloc[-1:]
                rf_pred = rf_model.predict(X)[0]
                xgb_pred = xgb_model.predict(X)[0]
                predictions[coin] = {
                    'rf': 'rise' if rf_pred == 1 else 'fall',
                    'xgb': 'rise' if xgb_pred == 1 else 'fall'
                }
            return render_template('predictions.html', predictions=predictions, coin_map=coins)
        except Exception as e:
            print(f"Error in predictions route: {str(e)}")
            return f"Error in predictions route: {str(e)}", 500

    return app