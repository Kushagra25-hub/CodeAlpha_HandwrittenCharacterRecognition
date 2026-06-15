import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

x_train = idx2numpy.convert_from_file(
    "data/emnist-letters-train-images-idx3-ubyte"
)

y_train = idx2numpy.convert_from_file(
    "data/emnist-letters-train-labels-idx1-ubyte"
)

img = x_train[0]



plt.imshow(img, cmap="gray")
plt.title(f"Raw Label = {y_train[0]}")
plt.show()