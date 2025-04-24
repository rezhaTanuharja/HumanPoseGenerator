import matplotlib.pyplot as plt
import torch
from config import CONFIG
from diffusionmodels import manifolds

from humanposegenerator import models, pipelines, utilities


def main():
    parameters = CONFIG["velocimeter"]

    manifold = manifolds.rotationalgroups.SpecialOrthogonal3(
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    velocity = models.sequential.Assembly(parameters["model"])
    velocity.to(parameters["device"])

    checkpoint = torch.load(
        "humanposegenerator/checkpoints/velocimeter.pth",
        weights_only=True,
    )

    state_dict = checkpoint["model_state_dict"]
    module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    velocity.load_state_dict(module_state_dict)

    velocity.eval()

    position = utilities.sample.sample_uniform_so3(num_samples=3000).to(
        parameters["device"],
    )

    time = torch.tensor(
        [
            parameters["period"],
        ],
        device=parameters["device"],
        dtype=parameters["data_type"],
    )

    time_decrement = parameters["period"] / 1200

    encoder = pipelines.encoders.create_encoder(parameters)

    for i in range(1200):
        position = manifold.exp(
            position,
            abs(time_decrement) * velocity(position.view(3000, 9), encoder(time)),
        )

        if i % 50 == 49:
            axis_angle = (
                manifold.log(
                    torch.eye(
                        3,
                        device=parameters["device"],
                        dtype=parameters["data_type"],
                    ),
                    position,
                )
                .cpu()
                .detach()
                .numpy()
            )

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            for j in range(axis_angle.shape[0]):
                ax.scatter(
                    axis_angle[j][:, 0],
                    axis_angle[j][:, 1],
                    axis_angle[j][:, 2],
                    marker="x",
                    s=1,
                    alpha=0.2,
                )

                ax.set_xlim([-torch.pi, torch.pi])
                ax.set_ylim([-torch.pi, torch.pi])
                ax.set_zlim([-torch.pi, torch.pi])

                ax.set_box_aspect([1, 1, 1])

                plt.savefig(f"diffusion_{(i + 1):04}.png")
                plt.close()


if __name__ == "__main__":
    main()
