import os

import numpy as np
import torch
import torch.distributed
import torch.nn.utils.prune
from config import CONFIG
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from humanposegenerator import dataloaders, models, pipelines, utilities


def main(local_rank: int = 0, world_size: int = 1):
    dataset = TensorDataset(
        dataloaders.amass_loaders.load_and_combine_amass_poses(CONFIG),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        drop_last=True,
    )

    diffuser = pipelines.diffuser.generate_diffuser(CONFIG)
    velocity = pipelines.speedometer.generate_speedometer(CONFIG)
    encoder = pipelines.encoders.generate_encoder(CONFIG)

    model, weights_to_prune = models.modulators.generate_film_model(CONFIG)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    criterion = torch.nn.SmoothL1Loss()

    train_loses = np.zeros((CONFIG["num_epochs"],))

    for epoch in range(CONFIG["num_epochs"]):
        sampler.set_epoch(epoch)

        running_loss = 0.0

        pred_norm = 0.0
        ref_norm = 0.0

        num_batches = 0

        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = CONFIG["dropout_rate"](epoch)

        if epoch in CONFIG["epoch_for_pruning"]:
            torch.nn.utils.prune.global_unstructured(
                parameters=weights_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=CONFIG["prune_amount"],
            )

        for (batch,) in dataloader:
            time = utilities.samplers.latin_hypercube_sampling(
                lower_bound=0.0,
                upper_bound=CONFIG["period"],
                num_samples=CONFIG["num_times"],
                device=CONFIG["device"],
            )

            noisy_poses, relative_poses = diffuser(batch, time)

            encoded_time = encoder(time)

            speed = velocity(
                relative_poses,
                encoded_time,
            ).flatten(-2)

            prediction = model(
                noisy_poses.flatten(-3),
                encoded_time,
            )

            optimizer.zero_grad()

            loss = criterion(prediction, speed)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            pred_norm += torch.norm(prediction).item()
            ref_norm += torch.norm(speed).item()
            num_batches += 1

        if local_rank == 0:
            print(
                f"Epoch {epoch + 1}, train loss: {(running_loss / num_batches):.4f}, pred / ref: {(pred_norm / num_batches):.1f} / {(ref_norm / num_batches):.1f}",
            )
            if not torch.isnan(loss).any():
                state_dict = model.state_dict()
                for name, module in model.named_modules():
                    if hasattr(module, "weight_orig"):
                        state_dict[f"{name}.weight_orig"] = module.weight_orig
                        state_dict[f"{name}.weight_mask"] = module.weight_mask

                checkpoint = {
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                torch.save(checkpoint, "humanposegenerator/checkpoints/human.pth")

            train_loses[epoch] = running_loss / num_batches

    if local_rank == 0:
        np.save("train_loses.npy", train_loses)


if __name__ == "__main__":
    torch.distributed.init_process_group("nccl")
    local_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    try:
        main(local_rank, world_size)
    except Exception as e:
        print(f"Received exception {e}")

    torch.distributed.destroy_process_group()
