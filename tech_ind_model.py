import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sklearn.preprocessing import MinMaxScaler
from typing import List
import tensorflow as tf
import numpy as np

from tensorflow import set_random_seed


from util import csv_to_dataset, get_best_model_path
from params import Parameters
import matplotlib.pyplot as plt

np.random.seed(4)
set_random_seed(4)


def evaluate(params, ohlcv_test, tech_ind_test, unscaled_y_test, y_normaliser: MinMaxScaler) -> None:
    """Evaluate model and give MSE + chart.

    Attrs:
        params:
        ohlcv_test:
        tech_ind_test:
        unscaled_y_test:
        y_normaliser:

    """
    model = tf.keras.models.load_model(get_best_model_path(params), compile=True)

    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    # Compute MSE
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print("Scaled MSE: ", scaled_mse)

    # Chart predictions
    plt.gcf().set_size_inches(22, 15, forward=True)
    start = 0
    end = -1
    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    # y_predicted = model.predict([ohlcv_histories, technical_indicators])
    # y_predicted = y_normaliser.inverse_transform(y_predicted)
    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])
    plt.show()


def main():
    """

    :return:
    """

    params = Parameters()
    if not params.SAVE_MODEL_PATH.exists():
        params.SAVE_MODEL_PATH.mkdir()

    print("Write params in JSON")
    json_string = Parameters().to_json()
    params_path = params.SAVE_MODEL_PATH / "params.json"
    with open(params_path.as_posix(), "w") as f:
        f.write(json_string)

    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
        params.CSV_DATA_PATH, params.num_history_points)

    print("----", ohlcv_histories.shape, technical_indicators.shape, next_day_open_values.shape)

    n = int(ohlcv_histories.shape[0] * params.train_split)

    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_true_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    # y_true_test = next_day_open_values[n:]
    unscaled_y_test = unscaled_y[n:]

    print("train_val split", ohlcv_train.shape)
    print("test split", ohlcv_test.shape)

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.TensorBoard(params.SAVE_MODEL_PATH.as_posix()),
        tf.keras.callbacks.ModelCheckpoint(
            get_best_model_path(params), monitor='val_loss', verbose=1, save_best_only=True,
        )
    ]

    # model = tech_net(technical_indicators.shape[1:], params)
    #
    # adam = optimizers.Adam(lr=params.LR)
    # model.compile(optimizer=adam, loss='mse')
    #
    # model.fit(x=[ohlcv_train, tech_ind_train], y=y_true_train, batch_size=params.BATCH_SIZE, epochs=params.EPOCHS,
    #           shuffle=True, validation_split=params.val_split_out_of_train,
    #           callbacks=callbacks)

    evaluate(params, ohlcv_test, tech_ind_test, unscaled_y_test, y_normaliser)


if __name__ == "__main__":
    main()
