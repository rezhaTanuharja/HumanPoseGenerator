import os
import random

import numpy as np
import torch
import torch.distributed
import torch.nn
import torch.optim
from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3

# from matplotlib import numpy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from humanposegenerator.models.modulators import FiLM
from pipelines.input import noiser
from pipelines.temporal import time_encoder
from utilities.initial_distributions import CheckerBoard


def set_seed(seed=42):
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed(seed)  # Set seed for CUDA
    torch.cuda.manual_seed_all(seed)  # Set seed for multi-GPU
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python’s built-in random
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = (
        False  # May slow down training but ensures reproducibility
    )


def latin_hypercube_sampling(num_samples: int, device: torch.device):
    intervals = torch.linspace(0, 1, num_samples + 1, device=device)[:-1]
    jitter = torch.rand(num_samples, device=device) / num_samples

    samples = intervals + jitter

    return samples


def configure_ddp():
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)


def main(local_rank: int):
    # set_seed(42)  # Call this before model initialization
    device = torch.device(f"cuda:{local_rank}")

    # num_samples = 600
    # num_times = 300
    # num_waves = 32
    # batch_size = 30

    num_samples = 1
    num_times = 600
    num_waves = 32
    batch_size = 1
    #
    device = torch.device("cuda", local_rank)

    manifold = SpecialOrthogonal3(data_type=torch.float32)
    manifold.to(device)

    initial_distribution = CheckerBoard()
    initial_distribution.to(device)

    # time = torch.tensor(
    #     [
    #         0.2,
    #         1.0,
    #         2.0,
    #     ]
    # )
    period = 1.6

    output_size = 3

    # initial_condition = manifold.exp(
    #     torch.eye(3, dtype=torch.float32, device=device),
    #     initial_distribution.sample(num_samples=num_samples),
    # )

    initial_condition = torch.eye(3, dtype=torch.float32, device=device).view(
        1, 1, 3, 3
    )

    if local_rank == 0:
        np.save("initial_condition.npy", initial_condition.cpu().detach().numpy())
    dataset = TensorDataset(initial_condition.squeeze(0))
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    checkpoint_path = "surrogate_point.pth"
    activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
    dropout_layer = torch.nn.Dropout(p=0.0)

    def initiate_component(component):
        for m in component.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight,
                    a=activation_layer.negative_slope,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    num_epochs = 1000000
    for epoch in range(num_epochs):
        torch.distributed.barrier()

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
            torch.nn.Linear(1 * 9, 80),
            # torch.nn.Linear(1 * 9, 16),
            activation_layer,
            dropout_layer,
            # torch.nn.Linear(16, 8),
            # torch.nn.Linear(16, 16),
            torch.nn.Linear(80, 80),
            activation_layer,
            # torch.nn.Linear(80, 80),
            # activation_layer,
            torch.nn.Linear(80, output_size),
        ).to(device)

        initiate_component(temporal_components)
        initiate_component(spatial_components)

        model = FiLM(modulator=temporal_components, head=spatial_components).to(device)

        model.train()

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

        # model = FiLMMergedMLP(2 * num_waves, 2 * 9, hidden_size, 3, device).to(device)

        # model.apply(apply_he)

        # model = SimpleNN(input_size, hidden_size, output_size)
        # model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        # model.load_state_dict(torch.load("model_state.pth"))

        # criterion = torch.nn.MSELoss()
        # criterion = torch.nn.HuberLoss(delta=1.5)
        # criterion = torch.nn.L1Loss()
        sec_criterion = torch.nn.SmoothL1Loss()
        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.1)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2
        )
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=lambda epoch: 0.95**epoch
        # )

        if os.path.exists("surrogate_point.pth"):
            map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=True
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        loss = torch.tensor(0)
        running_loss = 0.0
        batch_id = 0
        for _ in range(500):
            for sample_points in dataloader:
                # time = period * torch.rand((num_times,), device=device)
                min_time = 0.0
                time = min_time + (period - min_time) * latin_hypercube_sampling(
                    num_times, device=device
                )

                encoded_time = time_encoder(time, period, num_waves=num_waves)

                points = sample_points[0]
                # print(points.shape)

                final_condition, velocity = noiser(
                    initial_condition=points.unsqueeze(0),
                    num_samples=batch_size,
                    time=time,
                    # time=torch.arange(start=0.0, end=2.0, step=1.0),
                )

                # axis_angle = manifold.log(torch.eye(3, device=device), final_condition)

                encoded_time = encoded_time.unsqueeze(1)
                encoded_time = encoded_time.repeat(1, batch_size, 1)

                encoded_time_data = encoded_time.view(
                    num_times * batch_size, 2 * num_waves
                )

                # final_condition = manifold.log(
                #     torch.eye(3, device=device), final_condition
                # )

                final_condition_data = final_condition.view(
                    num_times * batch_size, 1 * 9
                )
                # input_data = torch.cat((encoded_time_data, final_condition_data), dim=1)
                velocity_data = velocity.view(num_times * batch_size, 3)

                # print("yes")

                optimizer.zero_grad()  # Zero out gradients
                # outputs = model(input_data)  # Forward pass
                # modulated_input = conditioner(final_condition_data, encoded_time_data)
                outputs = model(final_condition_data, encoded_time_data)
                # outputs = model(
                #     (encoded_time_data, final_condition_data)
                # )  # Forward pass
                crit = criterion(outputs, velocity_data)  # Compute loss
                # loss_alpha = 0.0
                # loss = (1.0 - loss_alpha) * crit + loss_alpha * crit / (
                #     torch.mean(torch.norm(velocity_data, p=2, dim=-1, keepdim=True))
                #     + 1e-8
                # )
                loss = crit

                loss.backward()  # Backpropagation
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()  # Update weights
                sec_loss = sec_criterion(outputs, velocity_data)

                running_loss += sec_loss.item()
                batch_id += 1

            scheduler.step()

        if local_rank == 0:
            print(
                f"Epoch {epoch + 1}, Loss: {loss.item():.6f}, Ref Loss: {(running_loss / batch_id):.6f}"
            )
            if not torch.isnan(loss).any():
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
            # torch.save(model.state_dict(), "model_state.pth")


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# for i in range(axis_angle.shape[0]):
#     ax.scatter(
#         axis_angle[i][:, 0],
#         axis_angle[i][:, 1],
#         axis_angle[i][:, 2],
#         marker="x",
#         alpha=0.4,
#     )
#
# ax.set_xlim([-torch.pi, torch.pi])
# ax.set_ylim([-torch.pi, torch.pi])
# ax.set_zlim([-torch.pi, torch.pi])
#
# ax.set_box_aspect([1, 1, 1])
#
# plt.show()

if __name__ == "__main__":
    configure_ddp()

    local_rank = torch.distributed.get_rank()
    # world_size = torch.distributed.get_world_size()

    main(local_rank)

    torch.distributed.destroy_process_group()
