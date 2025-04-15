import math
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.utils.prune
from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3

from main import set_seed
from models.modulation import FiLM
from pipelines.temporal import time_encoder
from utilities.initial_distributions import CheckerBoard


def sample_uniform_so3(n=1):
    """Generate n random rotations uniformly sampled from SO(3) using quaternions."""
    u1, u2, u3 = torch.rand(3, n)  # Sample three independent uniform variables

    # Compute quaternion components
    q0 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q1 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q2 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q3 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    # Form rotation matrices
    R = torch.stack(
        [
            1 - 2 * (q2**2 + q3**2),
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
            2 * (q1 * q2 + q0 * q3),
            1 - 2 * (q1**2 + q3**2),
            2 * (q2 * q3 - q0 * q1),
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            1 - 2 * (q1**2 + q2**2),
        ],
        dim=-1,
    ).reshape(n, 3, 3)

    return R


# Example usage: generate 5 random SO(3) matrices
# rotations = sample_uniform_so3(5)
# print(rotations)

num_samples = 10
num_times = 1600
num_waves = 32

period = 1.6

dt = period / num_times


device = torch.device("cuda")

checkpoint_path = "newly_trained.pth"
activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
dropout_layer = torch.nn.Dropout(p=0.0)

input_size = 9 + 2 * num_waves
hidden_size = 128
output_size = 3

temporal_components = torch.nn.Sequential(
    torch.nn.Linear(2 * num_waves, 32),
    activation_layer,
    torch.nn.Linear(32, 32),
    activation_layer,
    dropout_layer,
    torch.nn.Linear(32, 18),
    # torch.nn.Linear(32, 16),
    # torch.nn.Linear(32, 8),
    activation_layer,
    # dropout_layer,
).to(device)

spatial_components = torch.nn.Sequential(
    torch.nn.Linear(1 * 9, 60),
    # torch.nn.Linear(1 * 9, 16),
    activation_layer,
    dropout_layer,
    # torch.nn.Linear(16, 8),
    # torch.nn.Linear(16, 16),
    torch.nn.Linear(60, 60),
    activation_layer,
    # torch.nn.Linear(60, 60),
    # activation_layer,
    torch.nn.Linear(60, output_size),
).to(device)


model = FiLM(modulator=temporal_components, head=spatial_components).to(device)
parameters_to_prune = [
    # (temporal_components[0], "weight"),  # First Linear in temporal_components
    (temporal_components[2], "weight"),  # Second Linear in temporal_components
    (temporal_components[5], "weight"),  # Last Linear in temporal_components
    (spatial_components[0], "weight"),  # First Linear in spatial_components
    (spatial_components[3], "weight"),  # Second Linear in spatial_components
    (spatial_components[5], "weight"),  # Last Linear in spatial_components
]

# Apply global pruning (L1-based)
torch.nn.utils.prune.global_unstructured(
    parameters=parameters_to_prune,
    pruning_method=torch.nn.utils.prune.L1Unstructured,
    amount=0.0,  # 20% of the smallest weights will be pruned globally
)

# conditioner = FiLM(modulator=temporal_components).to(device)


# model = torch.nn.Sequential(
#     torch.nn.Linear(input_size, hidden_size, dtype=torch.float32),
#     activation_layer,
#     # dropout_layer,
#     torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float32),
#     activation_layer,
#     # dropout_layer,
#     # torch.nn.Linear(hidden_size, hidden_size),
#     # activation_layer,
#     # dropout_layer,
#     # torch.nn.Linear(hidden_size, hidden_size),
#     # activation_layer,
#     # dropout_layer,
#     torch.nn.Linear(hidden_size, output_size, dtype=torch.float32),
# ).to(device)

# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# model = SimpleNN(input_size, hidden_size, output_size)

checkpoint = torch.load(checkpoint_path, weights_only=True)

state_dict = checkpoint["model_state_dict"]
module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# model.load_state_dict(checkpoint["model_state_dict"])
model.load_state_dict(module_state_dict)
model.eval()
model.to(device)
for i, module in enumerate(temporal_components):
    if hasattr(module, "weight_mask"):  # Check if pruning was applied
        mask = module.weight_mask
        num_masked = torch.sum(mask == 0).item()
        total_params = mask.numel()
        sparsity = num_masked / total_params * 100
        print(f"Layer {i}: Pruned {sparsity:.2f}% ({num_masked}/{total_params})")

for i, module in enumerate(spatial_components):
    if hasattr(module, "weight_mask"):  # Check if pruning was applied
        mask = module.weight_mask
        num_masked = torch.sum(mask == 0).item()
        total_params = mask.numel()
        sparsity = num_masked / total_params * 100
        print(f"Layer {i}: Pruned {sparsity:.2f}% ({num_masked}/{total_params})")

initial_distribution = CheckerBoard()
initial_distribution.to(device)

manifold = SpecialOrthogonal3(data_type=torch.float32)
manifold.to(device)

# initial_condition = manifold.exp(
#     torch.eye(3, device=device, dtype=torch.float32),
#     initial_distribution.sample(num_samples=num_samples),
# )
#
set_seed(42)
final_condition = sample_uniform_so3(num_samples).to(device).unsqueeze(0)
time = torch.tensor(
    [
        1.6,
    ],
    dtype=torch.float32,
    device=device,
)


# final_condition, _ = noiser(
#     initial_condition=initial_condition,
#     num_samples=num_samples,
#     time=torch.tensor(
#         [
#             1.6,
#         ],
#         dtype=torch.float32,
#         device=device,
#     ),
# )


for j in range(num_times + 1):
    encoded_time = time_encoder(time, period, num_waves=num_waves)

    encoded_time = encoded_time.view(1, 2 * num_waves)
    encoded_time_data = encoded_time.repeat(num_samples, 1)
    final_condition_data = final_condition.view(num_samples, 9)

    input_data = torch.cat((encoded_time_data, final_condition_data), dim=1)

    # outputs = model(input_data)  # Forward pass

    outputs = model(final_condition_data, encoded_time_data)
    # outputs = spatial_components(modulated_input)
    # outputs = model((encoded_time_data, final_condition_data))  # Forward pass

    final_condition = manifold.exp(final_condition, 1.0 * dt * outputs)
    time = time - dt

    if j % 40 == 0:
        axis_angle = manifold.log(
            torch.eye(3, device=device, dtype=torch.float32),
        #     torch.tensor([
        #         [math.cos(math.pi * 0.7), 0.0, -math.sin(math.pi * 0.7)],
        #         [0.0, 1.0, 0.0],
        #         [math.sin(math.pi * 0.7), 0.0, math.cos(math.pi * 0.7)],
        # ], dtype=torch.float32, device=device),
            final_condition
        )
        axis_angle = axis_angle.cpu().detach().numpy()

        if j == 1600:
            numpy.save("file.npy", axis_angle)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(axis_angle.shape[0]):
            ax.scatter(
                axis_angle[i][:, 0],
                axis_angle[i][:, 1],
                axis_angle[i][:, 2],
                marker="x",
                s=5,
                alpha=0.2,
            )

            ax.set_xlim([-torch.pi, torch.pi])
            ax.set_ylim([-torch.pi, torch.pi])
            ax.set_zlim([-torch.pi, torch.pi])

            ax.set_box_aspect([1, 1, 1])

            plt.savefig(f"distribution_{j:04}.png")
            plt.close()
