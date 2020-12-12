from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import json
import argparse


API_KEY: str = "998C6OC5B8AWLSOQ"


def save_dataset(symbol, time_window, api_key: str = API_KEY):
    # credentials = json.load(open('creds.json', 'r'))
    # api_key = credentials['av_api_key']

    print(symbol, time_window)
    ts = TimeSeries(key=api_key, output_format='pandas')
    if time_window == 'intraday':
        data, meta_data = ts.get_intraday(
            symbol='MSFT', interval='1min', outputsize='full')
    elif time_window == 'daily':
        data, meta_data = ts.get_daily(symbol, outputsize='full')
    elif time_window == 'daily_adj':
        data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')

    pprint(data.head(10))

    path = f'/media/dorel/DATA/work/stock-trading-ml/data/{symbol}_{time_window}.csv'
    data.to_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbol', type=str, help="the stock symbol you want to download")
    parser.add_argument('--time_window', type=str, choices=[
                        'intraday', 'daily', 'daily_adj'],
                        help="the time period you want to download the stock history for")

    namespace = parser.parse_args()
    save_dataset(**vars(namespace))
