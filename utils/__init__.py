import matplotlib.pyplot as plt


def make_plot(hist):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
  ax1.plot(hist.history['accuracy'], label='accuracy')
  ax1.plot(hist.history['val_accuracy'], label='val_accuracy')
  ax2.plot(hist.history['loss'], label='loss')
  ax2.plot(hist.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax2.set_xlabel('Epoch')
  ax1.set_ylabel('Accuracy')
  ax2.set_ylabel('Loss')
  ax1.legend()
  ax2.legend()
