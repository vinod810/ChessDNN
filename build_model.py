import os
import tensorflow as tf
import numpy as np
from typing import Iterable, Tuple, List
from tensorflow.keras.callbacks import ModelCheckpoint


from prepare_data import BOARD_SHAPE, COMPRESSION, OUT_DIR
TANH_FACTOR = 200

def score_to_tf_tanh(score):
    return tf.tanh(score / TANH_FACTOR)

def tanh_to_score(tanh):
    if tanh > 0.999999:
        tanh = 0.999999
    if tanh < -0.999999:
        tanh = -0.999999
    return round(np.arctanh(tanh) * TANH_FACTOR)

# Define the path to save the best model
# TODO create directory if non-existent
MODEL_FILEPATH = "model/model.keras"  # or "best_model.h5" for HDF5 format

# ----- Parse TFRecord -----
def parser(example_proto):
    feature_spec = {
        "board": tf.io.FixedLenFeature([], tf.string),
        "score": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_spec)

    board_raw = tf.io.decode_raw(parsed["board"], out_type=tf.float32)
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
    )
    if shuffle:
        ds = ds.shuffle(50_000)

    ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ----- Train / Val split by files -----
def get_train_val_datasets(
    data_dir=OUT_DIR, batch_size=256, train_frac=0.9):

    files = sorted(tf.io.gfile.glob(os.path.join(data_dir, "*.tfrecord")))
    n_train = int(len(files) * train_frac)

    train_files = files[:n_train]
    val_files = files[n_train:]

    train_ds = load_dataset(train_files, batch_size=batch_size, shuffle=True)
    val_ds   = load_dataset(val_files, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds

if __name__ == '__main__':

    train_ds, val_ds = get_train_val_datasets(batch_size=128)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=BOARD_SHAPE,),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Input(shape=[128, 18, 8, 8]),

        #tf.keras.layers.Flatten(input_shape=[BOARD_SHAPE[0] * BOARD_SHAPE[1] * BOARD_SHAPE[2], 1])
        #layers.Input(shape=(x_train.shape[1],)),
        #tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        #tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation='linear')  # regression output
    ])

    model.summary()

    model.compile(optimizer="adam", loss="mae", metrics=['mae'])


    # Create a ModelCheckpoint callback
    # monitor: the metric to monitor (e.g., 'val_loss', 'val_accuracy')
    # save_best_only: if True, only saves when the monitored metric improves
    # mode: 'min' for metrics like loss (lower is better), 'max' for metrics like accuracy (higher is better)
    checkpoint = ModelCheckpoint(
        MODEL_FILEPATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1  # Displays messages when the callback takes an action
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )

    model.fit(train_ds,
              epochs=100,
              steps_per_epoch=10000,      # dataset is infinite stream
              validation_data=val_ds,
              validation_steps=100,
              callbacks=[checkpoint, early_stopping_callback]
              )

    pairs_printed = 0
    for batch_x, batch_y in val_ds:  # your tf.data.Dataset
        y_pred = model.predict(batch_x)

        for t, p in zip(batch_y.numpy(), y_pred.flatten()):  # pred_labels):
            try:
                print(f"{tanh_to_score(t)}, {tanh_to_score(p)}")
            except (ValueError, OverflowError):
                print("NA, NA")
                pass
            pairs_printed += 1
            if pairs_printed >= 20:
                break
        if pairs_printed >= 20:
            break




