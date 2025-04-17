import os

import torch
import torch.distributed
from config import CONFIG
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from humanposegenerator import models, pipelines, utilities


def main(local_rank: int = 0, world_size: int = 1):
    dataset = utilities.load_amass.load_and_combine_amass_poses(
        CONFIG["data_directory"],
        CONFIG["joint_indices"],
    )

    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset).to(dtype=CONFIG["data_type"]),
    )

    sampler = DistributedSampler(
        tensor_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        tensor_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        drop_last=True,
    )

    diffuser = pipelines.diffuser.create_diffuser(CONFIG)
    velocimeter = pipelines.velocimeter.create_velocimeter(CONFIG)
    encoder = pipelines.encoders.create_encoder(CONFIG)

    model = models.sequential.Assembly(CONFIG["model"])
    model.to(CONFIG["device"])
    model.train()

    distributed_model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        running_loss = 0.0
        num_batches = 0

        utilities.torch_module.set_dropout_rate(
            model,
            CONFIG["dropout_rate"](epoch),
        )

        for batch in dataloader:
            (poses,) = batch

            poses = poses.to(CONFIG["device"])

            time = utilities.samplers.latin_hypercube_sampling(
                lower_bound=0.0,
                upper_bound=CONFIG["period"],
                num_samples=CONFIG["num_times"],
                device=CONFIG["device"],
            )

            noisy_poses, relative_poses = diffuser(poses, time)

            encoded_time = encoder(time)

            predicted_velocity = distributed_model(
                noisy_poses.flatten(-3),
                encoded_time,
            )

            reference_velocity = velocimeter(
                relative_poses,
                encoded_time,
            ).flatten(-2)

            optimizer.zero_grad()

            loss = criterion(predicted_velocity, reference_velocity)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        if local_rank == 0:
            print(f"Epoch {epoch + 1}, losses: {(running_loss / num_batches):.4f}")

            state_dict = utilities.torch_module.get_state_dict(model)

            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(checkpoint, CONFIG["checkpoint"])


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")

    local_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    main(local_rank, world_size)

    torch.distributed.destroy_process_group()
