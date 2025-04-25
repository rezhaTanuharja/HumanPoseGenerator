"""The config file for training a velocimeter and a pose generator."""

import torch

CONFIG = {}

activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)

CONFIG["velocimeter"] = {
    "device": torch.device("cuda"),
    "data_type": torch.float32,
    "alpha": 0,
    "period": 1.6,
    "num_waves": 8192,
    "num_iterations": 24,
    "mean_squared_displacement": lambda t: 1.5 * t**4,
    "num_frequencies": 32,
    "batch_size": 1,
    "num_times": 600,
    "num_epochs": 8000,
    "learning_rate": 5e-5,
    "dropout_rate": 0.0,
    "checkpoint": "humanposegenerator/checkpoints/velocimeter.pth",
}

CONFIG["velocimeter"]["model"] = [
    {
        "modulator": {
            "signal_shapes": (
                2 * CONFIG["velocimeter"]["num_frequencies"],
                32,
                32,
                18,
            ),
            "dropout_at": (),
            "activation_layer": activation_layer,
        },
    },
    {
        "mlp": {
            "signal_shapes": (9, 32, 32, 3),
            "dropout_at": (),
            "activation_layer": activation_layer,
            "drop_last_activation": True,
        },
    },
]

CONFIG["pose_generator"] = {
    "device": torch.device("cuda"),
    "data_type": torch.float32,
    "alpha": 2,
    "period": 1.6,
    "num_waves": 1024,
    "num_iterations": 10,
    "mean_squared_displacement": lambda t: 1.5 * t**4,
    "num_frequencies": 32,
    "batch_size": 4,
    "total_size": -1,
    "num_times": 100,
    "num_epochs": 750,
    "learning_rate": 0.00003,
    "dropout_rate": lambda epoch: max(0.0, 0.4 - 0.0008 * epoch),
    "checkpoint": "humanposegenerator/checkpoints/pose_generator.pth",
    "velocity_checkpoint": "humanposegenerator/checkpoints/velocimeter.pth",
    "joint_indices": list(range(3, 66)),
    "epoch_for_pruning": [120, 240, 360, 420, 480, 540, 700],
    "prune_amount": 0.125,
}

CONFIG["pose_generator"]["data_directory"] = [
    "/home/ratanuh/Datasets/ACCAD/Male2MartialArtsExtended_c3d/",
]

CONFIG["pose_generator"]["num_joints"] = (
    len(CONFIG["pose_generator"]["joint_indices"]) // 3
)

CONFIG["pose_generator"]["velocimeter"] = CONFIG["velocimeter"]["model"]

CONFIG["pose_generator"]["model"] = [
    {
        "modulator": {
            "signal_shapes": (
                2 * CONFIG["pose_generator"]["num_frequencies"],
                64,
                64,
                2 * CONFIG["pose_generator"]["num_joints"] * 9,
            ),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
        },
    },
    {
        "mlp": {
            "signal_shapes": (CONFIG["pose_generator"]["num_joints"] * 9, 256, 256),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
            "drop_last_activation": True,
        },
    },
    {
        "modulator": {
            "signal_shapes": (
                2 * CONFIG["pose_generator"]["num_frequencies"],
                64,
                64,
                2 * 256,
            ),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
        },
    },
    {
        "mlp": {
            "signal_shapes": (256, CONFIG["pose_generator"]["num_joints"] * 3),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
            "drop_last_activation": True,
        },
    },
]
