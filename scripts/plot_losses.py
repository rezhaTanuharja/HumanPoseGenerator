"""A script to plot training and validation losses over number of epochs."""

import matplotlib.pyplot as plt
import numpy as np

losses_1 = np.load("losses/mae_no_dropout_higher_lr.npy")
losses_2 = np.load("losses/mae_gradual_dropout_higher_lr.npy")
losses_3 = np.load("losses/mae_gradual_dropout_higher_higher_lr.npy")
losses_4 = np.load("l1_losses.npy")
# losses_4 = np.load("losses/l1_dropout_40_train_losses.npy")
# losses_1 = np.load("losses/l2_dropout_40_full_accad.npy")
# losses_2 = np.load("losses/l1_dropout_20_train_losses.npy")
# losses_1 = np.load("losses/mae_rmse_gradual_dropout.npy")
# losses_2 = np.load("losses/mae_mse_gradual_dropout.npy")
# losses_2 = np.load("losses/mae_more_gradual_dropout.npy")
# losses_5 = np.load("losses/mae_no_dropout.npy")
# losses_3 = np.load("losses/mae_gradual_dropout.npy")
# losses_1 = np.load("losses/mae_gradual_dropout_high_lr.npy")
# losses_2 = np.load("losses/mae_gradual_dropout_scheduler.npy")
# losses_4 = np.load("losses/mae_step_dropout.npy")
# losses_7 = np.load("losses/mae_gradual_dropout_higher_lr_decay.npy")
# losses_8 = np.load("losses/mae_curriculum_dropout_higher_lr.npy")
# losses_9 = np.load("losses/mae_gradual_dropout_higher_lr_scheduler.npy")
# losses_2 = np.load("losses/combined_rmse_losses_gradual_dropout.npy")
# losses_3 = np.load("losses/separate_mae_losses_gradual_dropout.npy")
# losses_4 = np.load("losses/separate_rmse_losses_gradual_dropout.npy")
# losses_4 = np.load("separate_l2_losses.npy")
# losses_5 = np.load("losses/l1_dropout_40_train_losses.npy")

num_epochs = 1 + np.arange(start=0, stop=losses_1.shape[0])
# num_epochs = 0 + np.arange(start=0, stop=400)

plt.plot(
    num_epochs,
    losses_1,
    label="No Dropout",
    color="black",
)

plt.plot(
    num_epochs,
    losses_2,
    label="Gradual Dropout",
    color="red",
)

plt.plot(
    num_epochs,
    losses_3,
    label="Separate: L1 Losses",
    color="blue",
)

plt.plot(
    num_epochs,
    losses_4,
    label="Separate: L2 Losses",
    color="green",
)

# plt.plot(
#     num_epochs,
#     losses_5,
#     label="60% Dropout",
#     color="orange",
# )

# plt.plot(
#     num_epochs,
#     losses_6,
#     label="60% Dropout",
#     color="purple",
# )

# plt.plot(
#     num_epochs,
#     losses_7,
#     label="60% Dropout",
#     color="brown",
# )

# plt.plot(
#     num_epochs,
#     losses_8,
#     label="60% Dropout",
#     color="cyan",
# )

# plt.plot(
#     num_epochs,
#     losses_9,
#     label="60% Dropout",
#     color="black",
# )

plt.xlim([0, 800])
plt.ylim([0.0, 0.5])
plt.xlabel("Number of Epochs")
plt.ylabel("Smooth L1 Losses")
plt.grid()
plt.legend()


plt.show()
