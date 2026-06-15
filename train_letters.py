import idx2numpy
import matplotlib.pyplot as plt

x_train = idx2numpy.convert_from_file(
    "data/emnist-letters-train-images-idx3-ubyte"
)

y_train = idx2numpy.convert_from_file(
    "data/emnist-letters-train-labels-idx1-ubyte"
)

plt.figure(figsize=(10,5))

for i in range(10):

    plt.subplot(2,5,i+1)

    plt.imshow(
        x_train[i],
        cmap="gray"
    )

    plt.title(
        y_train[i]
    )

    plt.axis("off")

plt.show()