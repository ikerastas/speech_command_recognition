import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from utils import make_plot
from utils.ds_utils import split_dataset, to_float, to_spectrogram, expand_dims
from utils.models import create_cnn_model

dataset_name = "speech_commands"
ds, info = tfds.load(name=dataset_name, with_info=True, as_supervised=True, try_gcs=True, split='train+test+validation')
num_classes = info.features["label"].num_classes

split_percentage = 0.15
base_train_ds, base_test_ds, base_val_ds = split_dataset(ds, split_percentage)

train_ds = base_train_ds.shuffle(128 * 128)
train_ds = train_ds.map(to_float).map(to_spectrogram).map(expand_dims).cache()

test_ds = base_test_ds.shuffle(128 * 128)
test_ds = test_ds.map(to_float).map(to_spectrogram).map(expand_dims).cache()

val_ds = base_val_ds.shuffle(128 * 128)
val_ds = val_ds.map(to_float).map(to_spectrogram).map(expand_dims).cache()

batch_size = 256

train_ds_batched = train_ds.padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_ds_batched = test_ds.padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_ds_batched = val_ds.padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

input_shape = (*to_spectrogram(np.zeros(16000), "")[0].shape, 1)
print(input_shape)

epochs = 1000
callbacks = [
  tf.keras.callbacks.ReduceLROnPlateau(patience=4, min_delta=1e-3, min_lr=1e-4, verbose=1),
  tf.keras.callbacks.EarlyStopping(min_delta=1e-4, patience=10, verbose=1)
]

dims = [32, 64, 128, 256, 512]
pooling_type = "max_pooling"  # ["max_pooling","average_pooling"]
flatten = True
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)

model = create_cnn_model(dims=dims,
                         pooling_type=pooling_type,
                         flatten=flatten,
                         input_shape=input_shape,
                         classes=num_classes,
                         optimizer=optimizer)

model_history = model.fit(train_ds_batched, validation_data=val_ds_batched, epochs=epochs, callbacks=callbacks)
make_plot(model_history)
test_loss, test_acc = model.evaluate(test_ds_batched, verbose=2)
