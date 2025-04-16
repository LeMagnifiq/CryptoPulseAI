from flask import Flask, render_template
from pycoingecko import CoinGeckoAPI

def create_app():
    app = Flask(__name__, template_folder='templates')  # Explicitly set template folder
    client = CoinGeckoAPI()

    @app.route('/')
    def home():
        return 'Hello, CryptoPulseAI!'

    @app.route('/prices')
    def prices():
        coins = ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin']
        try:
            market_data = client.get_coins_markets(
                vs_currency='usd',
                ids=coins,
                order='market_cap_desc',
                per_page=100,
                page=1,
                sparkline=False,
                price_change_percentage='24h'
            )
            print("API Response:", market_data)
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
            return render_template('prices.html', prices=formatted_data)
        except Exception as e:
            print(f"Error in prices route: {str(e)}")
            return f"Error in prices route: {str(e)}", 500

    return app