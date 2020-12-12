import pandas as pd
from sklearn import preprocessing
import numpy as np
import os

from params import Parameters


def csv_to_dataset(csv_path: str, num_history_points: int = 50):
    """

    csv_path:
    num_history_points:

    :return:
    """

    data = pd.read_csv(csv_path)
    # Drop the date
    data = data.drop('date', axis=1)
    # Drop the titles
    data = data.drop(0, axis=0)
    data = data.values

    # COLUMNS [open, high, low, close, volume]
    data_normaliser = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_normalised = data_normaliser.fit_transform(data)

    # using the last {history_points} open close high low volume data points, predict the next open value
    # SHAPE: [N, num_history_points, 5]
    ohlcv_histories_normalised = np.array([data_normalised[i:i + num_history_points].copy()
                                           for i in range(len(data_normalised) - num_history_points)])

    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + num_history_points].copy()
                                                for i in range(len(data_normalised) - num_history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, axis=-1)

    next_day_open_values = np.array([data[:, 0][i + num_history_points].copy()
                                     for i in range(len(data) - num_history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        technical_indicators.append(np.array([sma]))
        # TODO: check if second tech indicator helps
        # macd = calc_ema(his, 12) - calc_ema(his, 26)
        # technical_indicators.append(np.array([sma, macd,]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    # check that input and output number of examples is the same
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] \
           == technical_indicators_normalised.shape[0]

    # Ground Truth, in range [0, 1]
    y_true = next_day_open_values_normalised

    return ohlcv_histories_normalised, technical_indicators_normalised, y_true, next_day_open_values, y_normaliser


def multiple_csv_to_dataset(test_set_name):
    """

    :param test_set_name:
    :return:
    """
    ohlcv_histories = 0
    technical_indicators = 0
    next_day_open_values = 0
    for csv_file_path in list(filter(lambda x: x.endswith('daily.csv'), os.listdir('./'))):
        if not csv_file_path == test_set_name:
            print(csv_file_path)
            if type(ohlcv_histories) == int:
                ohlcv_histories, technical_indicators, next_day_open_values, _, _ = csv_to_dataset(csv_file_path)
            else:
                a, b, c, _, _ = csv_to_dataset(csv_file_path)
                ohlcv_histories = np.concatenate((ohlcv_histories, a), 0)
                technical_indicators = np.concatenate((technical_indicators, b), 0)
                next_day_open_values = np.concatenate((next_day_open_values, c), 0)

    ohlcv_train = ohlcv_histories
    tech_ind_train = technical_indicators

    # Ground Truth
    y_true = next_day_open_values

    ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser = csv_to_dataset(test_set_name)

    return ohlcv_train, tech_ind_train, y_true, ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser


def get_best_model_path(params: Parameters) -> str:
    checkpoint_path = params.SAVE_MODEL_PATH / "best_model.h5"

    return checkpoint_path.as_posix()