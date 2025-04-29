"""The config file for training a velocimeter and a pose generator."""

import torch
from torch.optim.optimizer import DeviceDict

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
    "batch_size": 8,
    "total_size": 400,
    "num_times": 250,
    "num_epochs": 2000,
    "learning_rate": 0.0010,
    "dropout_rate": lambda epoch: max(0.0, 0.3 - 0.00025 * epoch),
    "checkpoint": "humanposegenerator/checkpoints/pose_generator.pth",
    "velocity_checkpoint": "humanposegenerator/checkpoints/velocimeter.pth",
    "joint_indices": list(range(3, 66)),
    "epoch_for_pruning": [(i + 1) * 80 for i in range(20)],
    "prune_amount": 0.00375,
}

CONFIG["pose_generator"]["data_directory"] = [
    "/home/ratanuh/Datasets/ACCAD/Male2MartialArtsExtended_c3d/",
]

CONFIG["pose_generator"]["num_joints"] = (
    len(CONFIG["pose_generator"]["joint_indices"]) // 3
)

CONFIG["pose_generator"]["edge_index"] = torch.tensor(
    [
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 4],
        [2, 0],
        [2, 1],
        [2, 5],
        [3, 0],
        [3, 6],
        [4, 1],
        [4, 7],
        [5, 2],
        [5, 8],
        [6, 3],
        [6, 9],
        [7, 4],
        [7, 10],
        [8, 5],
        [9, 6],
        [10, 7],
        [11, 14],
        [12, 15],
        [13, 16],
        [14, 11],
        [15, 12],
        [15, 17],
        [16, 13],
        [16, 18],
        [17, 15],
        [17, 19],
        [18, 16],
        [18, 20],
        [19, 17],
        [20, 18],
    ],
    device=CONFIG["pose_generator"]["device"],
    dtype=torch.long,
).t()

CONFIG["pose_generator"]["velocimeter"] = CONFIG["velocimeter"]["model"]

# CONFIG["pose_generator"]["model"] = [
#     {
#         "modulator": {
#             "signal_shapes": (
#                 2 * CONFIG["pose_generator"]["num_frequencies"],
#                 48,
#                 48,
#                 2 * CONFIG["pose_generator"]["num_joints"] * 9,
#             ),
#             "dropout_at": (1,),
#             "activation_layer": activation_layer,
#         },
#     },
#     {
#         "gnn": {
#             "signal_shapes": (9, 9),
#             "activation_layer": activation_layer,
#             "edge_index": CONFIG["pose_generator"]["edge_index"],
#             "drop_last_activation": True,
#         },
#     },
#     {
#         "modulator": {
#             "signal_shapes": (
#                 2 * CONFIG["pose_generator"]["num_frequencies"],
#                 48,
#                 48,
#                 2 * CONFIG["pose_generator"]["num_joints"] * 9,
#             ),
#             "dropout_at": (1,),
#             "activation_layer": activation_layer,
#         },
#     },
#     {
#         "mlp": {
#             "signal_shapes": (
#                 CONFIG["pose_generator"]["num_joints"] * 9,
#                 512,
#                 512,
#                 512,
#                 CONFIG["pose_generator"]["num_joints"] * 3,
#             ),
#             "dropout_at": (0,),
#             "activation_layer": activation_layer,
#             "drop_last_activation": True,
#         },
#     },
# ]

CONFIG["pose_generator"]["model"] = [
    {
        "modulator": {
            "signal_shapes": (
                2 * CONFIG["pose_generator"]["num_frequencies"],
                48,
                48,
                2 * CONFIG["pose_generator"]["num_joints"] * 9,
            ),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
        },
    },
    {
        "gnn": {
            "signal_shapes": (9, 9),
            "activation_layer": activation_layer,
            "edge_index": CONFIG["pose_generator"]["edge_index"],
            "drop_last_activation": True,
        },
    },
    # {
    #     "modulator": {
    #         "signal_shapes": (
    #             2 * CONFIG["pose_generator"]["num_frequencies"],
    #             48,
    #             48,
    #             2 * CONFIG["pose_generator"]["num_joints"] * 9,
    #         ),
    #         "dropout_at": (1,),
    #         "activation_layer": activation_layer,
    #     },
    # },
    {
        "concatenator": {
            "signal_shapes": (
                CONFIG["pose_generator"]["num_joints"] * 9
                + 2 * CONFIG["pose_generator"]["num_frequencies"],
                512,
                512,
                512,
                CONFIG["pose_generator"]["num_joints"] * 3,
            ),
            "dropout_at": (0,),
            "activation_layer": activation_layer,
            "drop_last_activation": True,
        },
    },
]
