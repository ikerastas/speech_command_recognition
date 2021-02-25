import tensorflow as tf
import tensorflow.keras.layers as tfl


def create_rnn_model(rnn_type, dims, bidirectional, input_dim, classes, optimizer, line_length=100):
  rnn_type = rnn_type.lower()
  if rnn_type == "lstm":
    RnnLayer = tfl.LSTM
  elif rnn_type == "gru":
    RnnLayer = tfl.GRU
  else:
    print("Wrong rnn_type!")
    return

  # create name
  dims_str = [str(dim) for dim in dims]

  if bidirectional:
    name = rnn_type + "_bi" + "_model_" + "_".join(dims_str)
  else:
    name = rnn_type + "_model_" + "_".join(dims_str)

  model = tf.keras.Sequential(name=name)
  model.add(tfl.Input(shape=(None, input_dim)))

  if bidirectional:
    if len(dims) == 1:
      model.add(tfl.Bidirectional(RnnLayer(dims[0], return_sequences=False), name="bidirectional_" + rnn_type))
    else:
      for index, dim in enumerate(dims[:-1]):
        model.add(tfl.Bidirectional(RnnLayer(dim, return_sequences=True), name="bidirectional_" + rnn_type + "_" + str(index)))
      model.add(tfl.Bidirectional(RnnLayer(dims[-1], return_sequences=False), name="bidirectional_" + rnn_type + "_" + str(len(dims) - 1)))
  else:
    if len(dims) == 1:
      model.add(RnnLayer(dims[0], return_sequences=False, name=rnn_type))
    else:
      for index, dim in enumerate(dims[:-1]):
        model.add(RnnLayer(dim, return_sequences=True, name=rnn_type + "_" + str(index)))
      model.add(RnnLayer(dims[-1], return_sequences=False, name=rnn_type + "_" + str(len(dims) - 1)))

  model.add(tfl.Dense(classes, activation='softmax', name="output_dense"))
  model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
  model.summary(line_length=line_length)
  return model


def create_cnn_model(dims, pooling_type, flatten, input_shape, classes, optimizer, line_length=100):
  pooling_type = pooling_type.lower()
  if pooling_type == "max_pooling":
    PoolingLayer = tfl.MaxPooling2D
  elif pooling_type == "average_pooling":
    PoolingLayer = tfl.AveragePooling2D
  else:
    print("Wrong pooling_type!")
    return

  # create name
  dims_str = [str(dim) for dim in dims]
  name = "cnn" + "_model_" + "_".join(dims_str)

  model = tf.keras.Sequential(name=name)
  model.add(tfl.Input(shape=input_shape))

  for index, dim in enumerate(dims):
    model.add(tfl.Conv2D(filters=dim, kernel_size=(3, 3), padding='same', activation="relu", name="conv2d" + "_" + str(index)))
    model.add(tfl.BatchNormalization())
    model.add(PoolingLayer())

  if flatten:
    model.add(tfl.Flatten())
  else:
    model.add(tfl.GlobalMaxPooling2D())

  model.add(tfl.Dense(1024, activation='relu'))
  model.add(tfl.Dropout(0.2))
  model.add(tfl.Dense(512, activation='relu'))
  model.add(tfl.Dense(classes, activation='softmax', name="output_dense"))

  model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
  model.summary(line_length=line_length)
  return model
