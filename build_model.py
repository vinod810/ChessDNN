import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path

from prepare_data import COMPRESSION, OUT_DIR, SHARD_SIZE, TANH_SCALE, BOARD_SHAPE, MAX_SCORE, board_repr_to_fen

MAX_SHARDS = 1000 # Use a smaller number for quicker hyper param tuning
DATA_DIR = OUT_DIR
DATA_FILES = sorted(tf.io.gfile.glob(os.path.join(DATA_DIR, "*.tfrecord.zlib")))[:MAX_SHARDS]
TRAIN_FACTOR = 0.99 if MAX_SHARDS > 200 else (0.98 if MAX_SHARDS > 100
                                              else (0.95 if MAX_SHARDS > 10 else (0.90 if MAX_SHARDS > 5 else
                                                                            (0.80 if MAX_SHARDS > 2 else 0.50))))
N_TRAIN = int(len(DATA_FILES) * TRAIN_FACTOR)
curr_dir = Path(__file__).resolve().parent
DNN_MODEL_FILEPATH = curr_dir / 'model' / 'medium-relu-mae.keras'

BATCH_SIZE = 8192 # 256 * 4 # AVX2 CPU = 256
NUM_EPOCHS = 100
TANH_MAX = np.tanh(MAX_SCORE / TANH_SCALE)
TANH_MIN = -TANH_MAX

def score_to_tf_tanh(score):
    tanh = tf.tanh(score / TANH_SCALE)
    return tanh

def tanh_to_score(tanh):
    tanh = TANH_MIN if tanh < TANH_MIN else tanh
    tanh = TANH_MAX if tanh > TANH_MAX else tanh
    return round(np.arctanh(tanh) * TANH_SCALE)

# ----- Parse TFRecord -----
def parser(example_proto):
    feature_spec = {
        "board": tf.io.FixedLenFeature([], tf.string),
        "score": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_spec)

    board_raw = tf.io.decode_raw(parsed["board"], out_type=tf.uint8)
    board = tf.reshape(board_raw, BOARD_SHAPE)
    board = tf.cast(board, tf.float32)  # convert for DNN input
    score = score_to_tf_tanh(parsed["score"])

    return board, score


# ----- Create dataset loader -----
def load_dataset(tfrecord_files, batch_size=256, shuffle=True):
    ds = tf.data.TFRecordDataset(
        tfrecord_files,
        compression_type=COMPRESSION,
        num_parallel_reads=tf.data.AUTOTUNE
    ) .repeat(count=NUM_EPOCHS)

    if shuffle:
        ds = ds.shuffle(1_000_000) #BATCH_SIZE * BATCH_SIZE)

    ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ----- Train / Val split by files -----
def get_train_val_datasets(batch_size=256):

    train_files = DATA_FILES[:N_TRAIN]
    val_files = DATA_FILES[N_TRAIN:]

    train = load_dataset(train_files, batch_size=batch_size, shuffle=True)
    val   = load_dataset(val_files, batch_size=batch_size, shuffle=False)

    return train, val

if __name__ == '__main__':

    train_ds, val_ds = get_train_val_datasets(batch_size=BATCH_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=BOARD_SHAPE,),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"), # TODO try - first layer tanh activation
        tf.keras.layers.Dense(256, activation="relu"), # Try SF 768->2x256 -> 32->32->1
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation='tanh')
    ])

    model.summary()

    model.compile(optimizer="adam", loss="mae", metrics=['mae']) # SF uses mse. Todo try mse

    os.makedirs(os.path.dirname(DNN_MODEL_FILEPATH), exist_ok=True)
    checkpoint = ModelCheckpoint(
        DNN_MODEL_FILEPATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1  # Displays messages when the callback takes an action
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )

    n_val = len(DATA_FILES) - N_TRAIN
    steps_per_epoch = N_TRAIN * int(SHARD_SIZE / BATCH_SIZE)
    validation_steps = n_val * int(SHARD_SIZE / BATCH_SIZE)
    print("steps_per_epoch", steps_per_epoch)
    print("validation_steps", validation_steps)

    model.fit(train_ds,
              epochs=NUM_EPOCHS,
              steps_per_epoch=steps_per_epoch,      # dataset is infinite stream
              validation_data=val_ds,
              validation_steps=validation_steps,
              callbacks=[checkpoint, early_stopping_callback]
              )

    pairs_printed = 0
    for batch_x, batch_y in val_ds:  # your tf.data.Dataset
        y_pred = model.predict(batch_x)

        for x, t, p in zip(batch_x.numpy(), batch_y.numpy(), y_pred.flatten()):  # pred_labels):
            try:
                fen = board_repr_to_fen(x.reshape(BOARD_SHAPE))
                if abs(tanh_to_score(t) - tanh_to_score(p)) > 200:
                    print(f"{tanh_to_score(t)}, {tanh_to_score(p)}, fen={fen}")
                    pairs_printed += 1
            except (ValueError, OverflowError):
                print("NA, NA")
                pass
            if pairs_printed >= 10:
                break
        if pairs_printed >= 10:
            break
