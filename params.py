from dataclasses import dataclass
from pathlib import Path

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Parameters:

    # DATA
    CSV_DATA_PATH: str = "/media/dorel/DATA/work/stock-trading-ml/data/msft_daily.csv"
    num_history_points: int = 50
    train_split = 0.9
    val_split_out_of_train = 0.1

    # TRAIN
    training_name: str = ""
    SAVE_MODEL_PATH = Path("/media/dorel/DATA/work/stock-trading-ml/experiments/2020_12_12_003/")

    EPOCHS: int = 200
    BATCH_SIZE: int = 64
    LR: float = 0.0002
