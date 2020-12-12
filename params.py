from pathlib import Path

# DATA
num_history_points: int = 50
train_split = 0.9
val_split_out_of_train = 0.1


# TRAIN
EPOCHS: int = 100
BATCH_SIZE: int = 64

LR: float = 0.0001

CSV_DATA_PATH: str = "/media/dorel/DATA/work/stock-trading-ml/data/msft_daily.csv"
SAVE_MODEL_PATH = Path("/media/dorel/DATA/work/stock-trading-ml/experiments/2020_12_12_002/")
