"""
Train a model to generate realistic human poses.

Author
------
Tanuharja, R.A. -- tanuharja@ias.uni-stuttgart.de

Date
----
2024-04-20
"""

import os

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from humanposegenerator import config, models, pipelines, utilities


def main(local_rank: int = 0, world_size: int = 1):
    pose_generator_config = config.CONFIG["pose_generator"]

    dataset = utilities.amass.load_and_combine_poses(
        pose_generator_config["data_directory"],
        pose_generator_config["joint_indices"],
    )

    tensor_dataset = TensorDataset(
        torch.from_numpy(dataset).to(dtype=pose_generator_config["data_type"]),
    )

    sampler = DistributedSampler(
        tensor_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        tensor_dataset,
        batch_size=pose_generator_config["batch_size"],
        sampler=sampler,
        drop_last=True,
    )

    diffuser = pipelines.diffuser.create_empirical_diffuser(pose_generator_config)
    velocimeter = pipelines.velocimeter.create_velocimeter(pose_generator_config)
    encoder = pipelines.encoders.create_encoder(pose_generator_config)

    model = models.sequential.Assembly(pose_generator_config["model"])
    model.to(pose_generator_config["device"])
    model.train()

    distributed_model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(
        distributed_model.parameters(),
        lr=pose_generator_config["learning_rate"],
    )

    criterion = torch.nn.MSELoss()

    for epoch in range(pose_generator_config["num_epochs"]):
        running_loss = 0.0
        num_batches = 0

        utilities.torch_module.set_dropout_rate(
            model,
            pose_generator_config["dropout_rate"](epoch),
        )

        for batch in dataloader:
            (poses,) = batch

            poses = poses.to(pose_generator_config["device"])

            time = utilities.sample.latin_hypercube_sampling(
                lower_bound=0.0,
                upper_bound=pose_generator_config["period"],
                num_samples=pose_generator_config["num_times"],
                device=pose_generator_config["device"],
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

            if torch.isnan(loss).any():
                continue

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

            torch.save(checkpoint, pose_generator_config["checkpoint"])


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")

    local_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    try:
        main(local_rank, world_size)
    except Exception as e:
        torch.distributed.destroy_process_group()
        raise e
