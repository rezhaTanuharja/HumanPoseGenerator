"""A script to plot training and validation losses over number of epochs."""

import matplotlib.pyplot as plt
import numpy as np

parallel_train_losses = np.load("parallel_train_losses.npy")
staggered_train_losses = np.load("staggered_train_losses.npy")
train_losses = np.load("train_loses.npy")
valid_losses = np.load("valid_loses.npy")

num_epochs = 1 + np.arange(start=0, stop=parallel_train_losses.shape[0])

plt.plot(
    num_epochs,
    parallel_train_losses,
    label="Parallel",
    color="black",
)

plt.plot(
    num_epochs,
    staggered_train_losses,
    label="Staggered",
    color="red",
)

plt.plot(
    num_epochs,
    train_losses,
    label="No Dropout",
    color="blue",
)

plt.plot(
    num_epochs,
    valid_losses,
    label="valid",
    color="green",
)

plt.xlabel("Number of Epochs")
plt.ylabel("L1 Losses")
plt.grid()
plt.legend()


plt.show()
