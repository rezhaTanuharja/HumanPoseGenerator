"""
Visualize a reverse pseudo-diffusion, i.e., from uniformly distributed points to the identity element.
Visualization is through 3D-plotting the axis-angle representation.
Saves several plots throughout the reverse diffusion process.

Author
------
Tanuharja, R.A. -- tanuharja@ias.uni-stuttgart.de

Date
----
2024-04-20
"""

import matplotlib.pyplot as plt
import torch
from diffusionmodels import manifolds

from humanposegenerator import config, models, pipelines, utilities


def main():
    parameters = config.CONFIG["velocimeter"]

    num_time_steps = 1200
    num_time_steps_per_save = 50
    num_samples = 3000

    manifold = manifolds.rotationalgroups.SpecialOrthogonal3(
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    velocity = models.sequential.Assembly(parameters["model"])
    velocity.to(parameters["device"])

    state_dict = utilities.torch_module.load_distributed_state_dict(
        "humanposegenerator/checkpoints/velocimeter.pth",
    )

    velocity.load_state_dict(state_dict)
    velocity.eval()

    position = utilities.sample.sample_uniform_so3(num_samples=num_samples).to(
        parameters["device"],
    )

    time = torch.tensor(
        [
            parameters["period"],
        ],
        device=parameters["device"],
        dtype=parameters["data_type"],
    )

    time_decrement = parameters["period"] / num_time_steps

    encoder = pipelines.encoders.create_encoder(parameters)

    for time_step in range(num_time_steps):
        position = manifold.exp(
            position,
            abs(time_decrement)
            * velocity(position.view(num_samples, 9), encoder(time)),
        )

        time = time - time_decrement

        if (time_step + 1) % num_time_steps_per_save == 0:
            axis_angle = (
                manifold.log(
                    torch.eye(
                        3,
                        device=parameters["device"],
                        dtype=parameters["data_type"],
                    ),
                    position,
                )
                .view(num_samples, 9)
                .cpu()
                .detach()
                .numpy()
            )

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                axis_angle[:, 0],
                axis_angle[:, 1],
                axis_angle[:, 2],
                marker="x",
                s=1,
                alpha=0.2,
            )

            ax.set_xlim([-torch.pi, torch.pi])
            ax.set_ylim([-torch.pi, torch.pi])
            ax.set_zlim([-torch.pi, torch.pi])

            ax.set_box_aspect([1, 1, 1])

            plt.savefig(f"diffusion_{(time_step + 1):04}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
