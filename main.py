import os

import numpy
import torch
import torch.distributed
import torch.nn.utils.prune
from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from models.modulators import FiLM
from pipelines.diffuser import generate_diffuser
from pipelines.encoders import time_encoder
from pipelines.speedometer import generate_speedometer
from utilities.samplers import latin_hypercube_sampling


def main(local_rank: int = 0):
    parameters = {
        "device": torch.device("cuda", local_rank),
        "data_type": torch.float32,
        "num_samples": 10,
        "num_waves": 4096,
        "num_sinusoids": 32,
        "mean_squared_displacement": lambda t: 1.5 * t**4,
        "alpha": 2,
        "num_iterations": 16,
        "velocity_checkpoint": "checkpoints/diffusion.pth",
        "batch_size": 2,
    }

    manifold = SpecialOrthogonal3(
        device=parameters["device"], data_type=parameters["data_type"]
    )

    dataset_directory = "/home/ratanuh/Datasets/"
    # dataset_name = "ACCAD/Male2Running_c3d/C6 - stand to run backward_poses.npz"
    dataset_name = "ACCAD/Male2Running_c3d/C18 - run to hop to walk_poses.npz"

    data = numpy.load(dataset_directory + dataset_name)
    axis_angle_poses = torch.tensor(
        data["poses"],
        device=parameters["device"],
        dtype=parameters["data_type"],
    )
    axis_angle_poses = axis_angle_poses.view(
        axis_angle_poses.shape[0], axis_angle_poses.shape[1] // 3, 3
    )

    print(axis_angle_poses.shape)

    poses = manifold.exp(
        torch.eye(3, device=parameters["device"], dtype=parameters["data_type"]),
        axis_angle_poses,
    )

    dataset = TensorDataset(poses)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=parameters["batch_size"],
        sampler=sampler,
    )

    diffuser = generate_diffuser(parameters)
    velocity = generate_speedometer(parameters)

    activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
    dropout_layer = torch.nn.Dropout(p=0.4)

    temporal_components = torch.nn.Sequential(
        torch.nn.Linear(2 * parameters["num_sinusoids"], 32),
        activation_layer,
        torch.nn.Linear(32, 32),
        activation_layer,
        dropout_layer,
        torch.nn.Linear(32, 52 * 18),
        activation_layer,
    ).to(parameters["device"])

    spatial_components = torch.nn.Sequential(
        torch.nn.Linear(52 * 9, 512),
        activation_layer,
        dropout_layer,
        torch.nn.Linear(512, 512),
        activation_layer,
        torch.nn.Linear(512, 52 * 3),
    ).to(parameters["device"])

    parameters_to_prune = [
        # (temporal_components[0], "weight"),  # First Linear in temporal_components
        (temporal_components[2], "weight"),  # Second Linear in temporal_components
        (temporal_components[5], "weight"),  # Last Linear in temporal_components
        (spatial_components[0], "weight"),  # First Linear in spatial_components
        (spatial_components[3], "weight"),  # Second Linear in spatial_components
        (spatial_components[5], "weight"),  # Last Linear in spatial_components
    ]

    torch.nn.utils.prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=0.0,  # 20% of the smallest weights will be pruned globally
    )

    model = FiLM(modulator=temporal_components, head=spatial_components)

    # checkpoint = torch.load("checkpoints/human.pth", weights_only=True)
    #
    # state_dict = checkpoint["model_state_dict"]
    # module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #
    # # model.load_state_dict(checkpoint["model_state_dict"])
    # model.load_state_dict(module_state_dict)
    model.train()

    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00010)
    criterion = torch.nn.SmoothL1Loss()

    for epoch in range(100000):
        sampler.set_epoch(epoch)

        running_loss = 0.0

        pred_norm = 0.0
        ref_norm = 0.0

        num_batches = 0

        if epoch in [150, 300, 450, 600, 750]:
            # apply_pruning(temporal_components, 0.15)
            # apply_pruning(spatial_components, 0.2)

            for layer in temporal_components:
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = layer.p - 0.075
            torch.nn.utils.prune.global_unstructured(
                parameters=parameters_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=0.10,
            )

        for batch in dataloader:
            clean_poses = batch[0]

            time = 1.6 * latin_hypercube_sampling(
                num_samples=100, device=parameters["device"]
            )

            noisy_poses = diffuser(clean_poses, time)

            diff = torch.einsum("...ji, ...jk -> ...ik", clean_poses, noisy_poses)

            encoded_time = time_encoder(
                time, period=1.6, num_waves=parameters["num_sinusoids"]
            )

            speed = velocity(
                # diff.flatten(-2).flatten(0, 1),
                diff.flatten(-2),
                encoded_time.unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, parameters["batch_size"], 52, 1),
            )

            prediction = model(
                noisy_poses.flatten(-3),
                encoded_time.unsqueeze(1).repeat(1, parameters["batch_size"], 1),
            )

            optimizer.zero_grad()

            loss = criterion(prediction, speed.flatten(-2))
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            pred_norm += torch.norm(prediction).item()
            ref_norm += torch.norm(speed.flatten(-2)).item()
            num_batches += 1

        if local_rank == 0:
            # print(
            #     f"Epoch {epoch + 1}, loss: {(running_loss / batch_id):.4f}, pred / ref: {(prediction_norm / batch_id):.1f} / {(reference_norm / batch_id):.1f}"
            # )
            print(
                f"Epoch {epoch + 1}, loss: {(running_loss / num_batches):.4f}, pred / ref: {(pred_norm / num_batches):.1f} / {(ref_norm / num_batches):.1f}"
            )
            # print(
            #     f"Epoch {epoch + 1}, Loss: {loss.item():.6f}, Ref Loss: {(running_loss / batch_id):.6f}"
            # )
            if not torch.isnan(loss).any():
                state_dict = model.state_dict()
                for name, module in model.named_modules():
                    if hasattr(module, "weight_orig"):  # If pruning was applied
                        state_dict[f"{name}.weight_orig"] = module.weight_orig
                        state_dict[f"{name}.weight_mask"] = module.weight_mask

                # for module in spatial_components.modules():
                #     if isinstance(module, torch.nn.Linear):
                #         torch.nn.utils.prune.remove(module, "weight")
                checkpoint = {
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    # "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(checkpoint, "checkpoints/human.pth")

        # print(diffuser)
        # print(velocity)


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    # local_rank = 0

    try:
        main(local_rank)
    except Exception as e:
        print(f"Received exception {e}")

    torch.distributed.destroy_process_group()
