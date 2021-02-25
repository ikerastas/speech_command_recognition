import tensorflow as tf
import tensorflow_datasets as tfds

from utils import make_plot
from utils.ds_utils import split_dataset, to_spectrogram, to_float
from utils.models import create_rnn_model

dataset_name = "speech_commands"
ds, info = tfds.load(name=dataset_name, with_info=True, as_supervised=True, try_gcs=True, split='train+test+validation', data_dir=".\\Dataset")
num_classes = info.features["label"].num_classes

split_percentage = 0.15
base_train_ds, base_test_ds, base_val_ds = split_dataset(ds, split_percentage)

train_ds = base_train_ds.shuffle(128 * 128)
train_ds = train_ds.map(to_float).map(to_spectrogram).cache()

val_ds = base_val_ds.shuffle(128 * 128)
val_ds = val_ds.map(to_float).map(to_spectrogram).cache()

test_ds = base_test_ds.shuffle(128 * 128)
test_ds = test_ds.map(to_float).map(to_spectrogram)

batch_size = 256

train_ds_batched = train_ds.padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_ds_batched = test_ds.padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_ds_batched = val_ds.padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

input_dim = train_ds_batched.element_spec[0].shape[2]
print(input_dim)

epochs = 1000
callbacks = [
  tf.keras.callbacks.ReduceLROnPlateau(patience=4, min_delta=1e-3, min_lr=1e-4, verbose=1),
  tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1e-4, verbose=1),
]

dims = [1024]
rnn_type = "lstm"  # ["gru","lstm"]
bidirectional = True
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model = create_rnn_model(rnn_type=rnn_type,
                         dims=dims,
                         bidirectional=bidirectional,
                         input_dim=input_dim,
                         classes=num_classes,
                         optimizer=optimizer)

model_history = model.fit(train_ds_batched, validation_data=val_ds_batched, epochs=epochs, callbacks=callbacks)

make_plot(model_history)
test_loss, test_acc = model.evaluate(test_ds_batched, verbose=2)
