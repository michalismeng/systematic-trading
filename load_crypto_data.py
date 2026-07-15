from alpha_vantage.cryptocurrencies import CryptoCurrencies
import pandas as pd
import os
# import matplotlib.pyplot as plt

def get_crypto_data(symbol, market='EUR'):
    key = os.getenv('API_KEY')
    cc = CryptoCurrencies(key=key, output_format='pandas')
    data, meta_data = cc.get_digital_currency_daily(symbol=symbol, market=market)
    data: pd.DataFrame = data
    data = data.rename(columns={ "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close", "5. volume": "volume" })
    # Sort by ascending date and discard last entry which is today (thus not yet completed)
    return data.sort_index()[:-1]

data = get_crypto_data('BTC')