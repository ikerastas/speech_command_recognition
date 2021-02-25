import tensorflow as tf
import tensorflow_io as tfio


def split_dataset(ds, val_test_percentage):
  if val_test_percentage < 0 or val_test_percentage > 1.0:
    print("val_test_percentage must be between (0,1)")
    return

  test_size = 1.0 - 2 * val_test_percentage

  full_ds_size = len(ds)
  train_ds_size = int(test_size * full_ds_size)
  val_ds_size = int(val_test_percentage * full_ds_size)

  ds = ds.shuffle(128 * 128, reshuffle_each_iteration=False)

  train_ds = ds.take(train_ds_size)
  remaining = ds.skip(train_ds_size)

  test_ds = remaining.take(val_ds_size)
  val_ds = remaining.skip(val_ds_size)

  print("Train size: {0}".format(len(train_ds)))
  print("Test size: {0}".format(len(test_ds)))
  print("Val size: {0}".format(len(val_ds)))

  return train_ds, test_ds, val_ds


def to_float(audio, label):
  audio = tf.cast(audio, tf.float32)
  audio = audio / 32768.0
  return audio, label


def to_spectrogram(audio, label):
  spectrogram = tfio.experimental.audio.spectrogram(audio, nfft=256, window=512, stride=128)
  return spectrogram, label


def expand_dims(spectrogram, label):
  spectrogram = tf.expand_dims(spectrogram, -1)
  return spectrogram, label
