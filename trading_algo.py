import numpy as np
import tensorflow as tf

from params import Parameters
from util import csv_to_dataset, get_best_model_path


def main():
    """TODO:

    """
    params = Parameters()
    model = tf.keras.models.load_model(get_best_model_path(params), compile=True)

    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = \
        csv_to_dataset(params.CSV_DATA_PATH, params.num_history_points)

    n = int(ohlcv_histories.shape[0] * params.train_split)
    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]

    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    buys = []
    sells = []
    # thresh = 0.7
    thresh = 0.1

    start = 0
    end = -1
    x = -1

    holding_stocks = False
    ratio_treshold = 0.005

    def compute_earnings(buys_, sells_):

        # purchase amount
        purchase_amt = 10
        stock = 0
        balance = 0
        while len(buys_) > 0 and len(sells_) > 0:
            if buys_[0][0] < sells_[0][0]:
                # time to buy $10 worth of stock
                balance -= purchase_amt
                stock += purchase_amt / buys_[0][1]
                buys_.pop(0)
            else:
                # time to sell all of our stock
                balance += stock * sells_[0][1]
                stock = 0
                sells_.pop(0)
        print(f"earnings: ${balance}")

    for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]])
        price_today = y_normaliser.inverse_transform(normalised_price_today)
        predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))

    ###### HIS CODE ########

        # delta = predicted_price_tomorrow - price_today
        # price_today = price_today[0][0]
        #
        # ratio = predicted_price_tomorrow / price_today
        # ratio_threshold = 0.02
        #
        # if ratio > 1 + ratio_threshold:
        #     buys.append((x, price_today))
        # elif ratio < 1 - ratio_threshold:
        #     sells.append((x, price_today))
        #
        # if delta > thresh and ratio > 1 + ratio_threshold:
        #     buys.append((x, price_today))
        # elif delta < -thresh and ratio < 1 - ratio_threshold:
        #     sells.append((x, price_today))

    ##### START STEFAN ######
        abs_price_today = price_today[0][0]

        if holding_stocks:
            price = buys[-1][1]
        else:
            price = abs_price_today

        # Absolute diff of price
        ratio = predicted_price_tomorrow / price

        if abs_price_today / price >= 1.3 and holding_stocks:
            sells.append((x, abs_price_today))
            holding_stocks = False

        elif ratio >= ratio_treshold and not holding_stocks:
            buys.append((x, abs_price_today))
            holding_stocks = True
            print("BUY: price today", abs_price_today, " predicted price tomorrow",
                  predicted_price_tomorrow)

        elif ratio < 1 - ratio_treshold and holding_stocks:
            sells.append((x, abs_price_today))
            holding_stocks = False

    ###################################################
        x += 1
    print(f"buys: {len(buys)}")
    print(f"sells: {len(sells)}")

    # we create new lists so we don't modify the original
    compute_earnings([b for b in buys], [s for s in sells])

    import matplotlib.pyplot as plt

    plt.gcf().set_size_inches(22, 15, forward=True)

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    if len(buys) > 0:
        plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
    if len(sells) > 0:
        plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

    # TRAIN TEST
    # ohlcv_train = ohlcv_histories[:n]
    # tech_ind_train = technical_indicators[:n]
    # y_train = next_day_open_values[:n]

    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])
    plt.show()


if __name__ == "__main__":
    main()
