# import torch.cuda
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils.prune
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors, show_image
from diffusionmodels import manifolds
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from humanposegenerator import config, models, pipelines, utilities


def main():
    parameters = config.CONFIG["pose_generator"]
    parameters["batch_size"] = 1

    num_time_steps = 1200

    num_poses = 200

    support_dir = "/home/ratanuh/Projects/HumanPoseGenerator/body_models"

    amass_npz_fname = (
        "/home/ratanuh/Datasets/ACCAD/Male2MartialArtsExtended_c3d/Extended 2_poses.npz"
    )

    bdata = np.load(amass_npz_fname)

    bm_fname = osp.join(
        support_dir,
        "smplh/male/model.npz",
    )
    dmpl_fname = osp.join(
        support_dir,
        "dmpls/male/model.npz",
    )

    num_betas = 16
    num_dmpls = 8

    manifold = manifolds.rotationalgroups.SpecialOrthogonal3(
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    pose_generator = models.sequential.Assembly(parameters["model"])

    parameters_to_prune = utilities.torch_module.get_parameters(
        pose_generator,
        "weight",
    )

    torch.nn.utils.prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=0.0,
    )

    state_dict = utilities.torch_module.load_distributed_state_dict(
        "humanposegenerator/checkpoints/pose_generator.pth",
    )

    pose_generator.load_state_dict(state_dict)
    pose_generator = pose_generator.to(parameters["device"])
    pose_generator.eval()

    time = torch.tensor(
        [
            parameters["period"],
        ],
        device=parameters["device"],
        dtype=parameters["data_type"],
    )

    time_decrement = parameters["period"] / num_time_steps

    encoder = pipelines.encoders.create_encoder(parameters)

    pose = utilities.sample.sample_uniform_so3(
        num_samples=num_poses * len(parameters["joint_indices"]) // 3,
    ).to(parameters["device"])

    pose = pose.view(num_poses, len(parameters["joint_indices"]) // 3, 3, 3)

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    for _ in range(num_time_steps):
        encoded_time = encoder(time).repeat(num_poses, 1, 1)

        pose = manifold.exp(
            pose,
            time_decrement
            * pose_generator(pose.flatten(-3), encoded_time.flatten(-2)).view(
                num_poses,
                len(parameters["joint_indices"]) // 3,
                3,
            ),
        )

        time = time - time_decrement

    axis_angle = manifold.log(
        torch.eye(
            3,
            device=parameters["device"],
            dtype=parameters["data_type"],
        ),
        pose,
    )

    bm = BodyModel(
        bm_fname=bm_fname,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname,
    ).to(parameters["device"])

    time_length = num_poses

    faces = c2c(bm.f)

    body_pose_beta = bm(
        pose_body=axis_angle.flatten(-2),
        betas=torch.tensor(
            np.repeat(
                bdata["betas"][:num_betas][np.newaxis],
                repeats=time_length,
                axis=0,
            ),
            device=parameters["device"],
            dtype=parameters["data_type"],
        ),
    )

    for i in range(num_poses):
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose_beta.v[i]),
            faces=faces,
            vertex_colors=np.tile(colors["grey"], (6890, 1)),
        )
        mv.set_static_meshes([body_mesh])
        body_image = mv.render(render_wireframe=False)
        show_image(body_image)
        plt.savefig(f"./output_files/martial_arts_{(i):04}.jpg")
        plt.close()


if __name__ == "__main__":
    main()
