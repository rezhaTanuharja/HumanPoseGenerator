import os

import torch
import torch.distributed
from config import CONFIG
from torch.nn.parallel import DistributedDataParallel

from humanposegenerator import models, pipelines, utilities


def main(local_rank: int = 0):
    velocimeter_config = CONFIG["velocimeter"]

    point_source = torch.eye(
        3,
        device=velocimeter_config["device"],
        dtype=velocimeter_config["data_type"],
    )
    point_source = point_source.view(1, 3, 3)
    point_source = point_source.repeat(velocimeter_config["batch_size"], 1, 1)

    diffuser = pipelines.diffuser.create_analytical_diffuser(velocimeter_config)
    encoder = pipelines.encoders.create_encoder(velocimeter_config)

    model = models.sequential.Assembly(velocimeter_config["model"])
    model.to(velocimeter_config["device"])
    model.train()

    distributed_model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(
        distributed_model.parameters(),
        lr=velocimeter_config["learning_rate"],
    )

    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
    )

    utilities.torch_module.set_dropout_rate(
        model,
        velocimeter_config["dropout_rate"],
    )

    for epoch in range(velocimeter_config["num_epochs"]):
        running_loss = 0.0
        num_reps = 0

        for _ in range(500):
            time = utilities.sample.latin_hypercube_sampling(
                lower_bound=0.0,
                upper_bound=velocimeter_config["period"],
                num_samples=velocimeter_config["num_times"],
                device=velocimeter_config["device"],
            )

            noisy_poses, reference_velocity = diffuser(point_source, time)

            encoded_time = encoder(time)

            predicted_velocity = distributed_model(
                noisy_poses.flatten(-2),
                encoded_time,
            )

            optimizer.zero_grad()

            loss = criterion(predicted_velocity, reference_velocity)
            loss.backward()

            running_loss += loss.item()
            num_reps += 1

        scheduler.step()

        if local_rank == 0:
            print(f"Epoch {epoch + 1}, losses: {(running_loss / num_reps):.4f}")

            state_dict = utilities.torch_module.get_state_dict(model)

            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(checkpoint, velocimeter_config["checkpoint"])

        optimizer.step()


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")

    local_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    main(local_rank)

    torch.distributed.destroy_process_group()
