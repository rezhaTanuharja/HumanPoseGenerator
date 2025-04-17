import torch

CONFIG = {}

CONFIG["device"] = torch.device("cuda")
CONFIG["data_type"] = torch.float32

# parameters for the diffusion process
CONFIG["alpha"] = 2
CONFIG["period"] = 1.6
CONFIG["num_waves"] = 1024
CONFIG["num_iterations"] = 10
CONFIG["mean_squared_displacement"] = lambda t: 1.5 * t**4

# the number of distinct frequencies used in time encoding
CONFIG["num_frequencies"] = 32

# parameters for datasets
CONFIG["data_directory"] = "/home/ratanuh/Datasets/ACCAD/Male2MartialArtsExtended_c3d/"
CONFIG["joint_indices"] = list(range(3, 66))
CONFIG["num_joints"] = len(CONFIG["joint_indices"]) // 3

# parameters for training
CONFIG["batch_size"] = 4
CONFIG["total_size"] = -1

CONFIG["num_times"] = 100

CONFIG["num_epochs"] = 750
CONFIG["learning_rate"] = 0.00003

CONFIG["epoch_for_pruning"] = [120, 240, 360, 420, 480, 540, 700]
CONFIG["prune_amount"] = 0.125

CONFIG["dropout_rate"] = lambda epoch: max(0.0, 0.4 - 0.0008 * epoch)

# checkpoint locations
CONFIG["checkpoint"] = "humanposegenerator/checkpoints/human.pth"
CONFIG["velocity_checkpoint"] = "humanposegenerator/checkpoints/diffusion.pth"

# configuration for model
activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)

CONFIG["model"] = [
    {
        "modulator": {
            "signal_shapes": (
                2 * CONFIG["num_frequencies"],
                64,
                64,
                2 * CONFIG["num_joints"] * 9,
            ),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
        },
    },
    {
        "mlp": {
            "signal_shapes": (CONFIG["num_joints"] * 9, 256, 256),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
            "drop_last_activation": True,
        },
    },
    {
        "modulator": {
            "signal_shapes": (
                2 * CONFIG["num_frequencies"],
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
            "signal_shapes": (256, CONFIG["num_joints"] * 3),
            "dropout_at": (1,),
            "activation_layer": activation_layer,
            "drop_last_activation": True,
        },
    },
]
