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
    num_time_steps_per_save = 10

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
        num_samples=len(parameters["joint_indices"]) // 3,
    ).to(parameters["device"])

    imw, imh = 1600, 1600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    for time_step in range(num_time_steps):
        encoded_time = encoder(time)

        pose = manifold.exp(
            pose,
            time_decrement
            * pose_generator(pose.flatten(), encoded_time.flatten()).view(
                len(parameters["joint_indices"]) // 3,
                3,
            ),
        )

        time = time - time_decrement

        if (time_step + 1) % num_time_steps_per_save == 0:
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

            time_length = len(bdata["trans"])

            faces = c2c(bm.f)

            body_pose_beta = bm(
                pose_body=axis_angle.flatten().view(1, 63).repeat(time_length, 1),
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

            body_mesh = trimesh.Trimesh(
                vertices=c2c(body_pose_beta.v[0]),
                faces=faces,
                vertex_colors=np.tile(colors["grey"], (6890, 1)),
            )
            mv.set_static_meshes([body_mesh])
            body_image = mv.render(render_wireframe=False)
            show_image(body_image)
            plt.savefig(
                f"./output_files/sampling/martial_arts_{(time_step + 1):04}.jpg"
            )
            plt.close()


if __name__ == "__main__":
    main()


# activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
# dropout_layer = torch.nn.Dropout(p=0.0)
#
# # temporal_components = torch.nn.Sequential(
# #     torch.nn.Linear(2 * 32, 32),
# #     activation_layer,
# #     torch.nn.Linear(32, 32),
# #     activation_layer,
# #     dropout_layer,
# #     torch.nn.Linear(32, 21 * 18),
# #     activation_layer,
# # ).to(device)
# #
# # spatial_components = torch.nn.Sequential(
# #     torch.nn.Linear(21 * 9, 512),
# #     activation_layer,
# #     dropout_layer,
# #     torch.nn.Linear(512, 512),
# #     activation_layer,
# #     # torch.nn.Linear(512, 512),
# #     # activation_layer,
# #     torch.nn.Linear(512, 21 * 3),
# # ).to(device)
# #
# # parameters_to_prune = [
# #     # (temporal_components[0], "weight"),  # First Linear in temporal_components
# #     (temporal_components[2], "weight"),  # Second Linear in temporal_components
# #     (temporal_components[5], "weight"),  # Last Linear in temporal_components
# #     (spatial_components[0], "weight"),  # First Linear in spatial_components
# #     (spatial_components[3], "weight"),  # Second Linear in spatial_components
# #     (spatial_components[5], "weight"),  # Last Linear in spatial_components
# #     # (spatial_components[12], "weight"),  # Last Linear in spatial_components
# # ]
#
# pre_modulator = torch.nn.Sequential(
#     torch.nn.Linear(
#         2 * 32,
#         64,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
#     torch.nn.Linear(
#         64,
#         64,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
#     torch.nn.Dropout(),
#     torch.nn.Linear(
#         64,
#         63 * 6,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
# ).to(device)
#
# pos_modulator = torch.nn.Sequential(
#     torch.nn.Linear(
#         2 * 32,
#         64,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
#     torch.nn.Linear(
#         64,
#         64,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
#     torch.nn.Dropout(),
#     torch.nn.Linear(
#         64,
#         2 * 512,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
# ).to(device)
#
#
# pre_block = torch.nn.Sequential(
#     torch.nn.Linear(
#         63 * 3,
#         512,
#     ),
#     torch.nn.LeakyReLU(negative_slope=0.005),
#     torch.nn.Dropout(),
#     torch.nn.Linear(
#         512,
#         512,
#     ),
# ).to(device)
#
# pos_block = torch.nn.Sequential(
#     torch.nn.LeakyReLU(negative_slope=0.005),
#     torch.nn.Linear(
#         512,
#         63,
#     ),
# ).to(device)
#
# parameters_to_prune = [
#     (pre_modulator[0], "weight"),
#     (pre_modulator[2], "weight"),
#     (pre_modulator[5], "weight"),
#     (pos_modulator[0], "weight"),
#     (pos_modulator[2], "weight"),
#     (pos_modulator[5], "weight"),
#     (pre_block[0], "weight"),
#     (pre_block[3], "weight"),
#     (pos_block[1], "weight"),
# ]
#
# torch.nn.utils.prune.global_unstructured(
#     parameters=parameters_to_prune,
#     pruning_method=torch.nn.utils.prune.L1Unstructured,
#     amount=0.0,  # 20% of the smallest weights will be pruned globally
# )
#
#
# # model = FiLM(modulator=temporal_components, main_block=spatial_components).to(device)
# model = NewFiLM(pre_modulator, pos_modulator, pre_block, pos_block).to(device)
#
# checkpoint = torch.load(
#     "humanposegenerator/checkpoints/pose_generator.pth",
#     weights_only=True,
# )
#
# state_dict = checkpoint["model_state_dict"]
# module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#
# # model.load_state_dict(checkpoint["model_state_dict"])
# model.load_state_dict(module_state_dict)
# model.eval()
#
# time = torch.tensor(
#     [
#         1.6,
#     ],
#     dtype=torch.float32,
#     device=device,
# )
#
# num_samples = 1
#
#
# support_dir = "/home/ratanuh/Datasets/"
#
# comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # comp_device = torch.device("cpu")
#
# amass_npz_fname = osp.join(
#     support_dir,
#     "ACCAD/Male2MartialArtsExtended_c3d/Extended 2_poses.npz",
# )  # the path to body data
# bdata = np.load(amass_npz_fname)
#
# # you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
# subject_gender = "male"
#
# print("Data keys available:%s" % list(bdata.keys()))
#
# print("The subject of the mocap sequence is  {}.".format(subject_gender))
#
# bm_fname = osp.join(
#     support_dir,
#     "/home/ratanuh/Projects/HumanFilter/amass/support_data/body_models/smplh/{}/model.npz".format(
#         subject_gender,
#     ),
# )
# dmpl_fname = osp.join(
#     support_dir,
#     "/home/ratanuh/Projects/HumanFilter/amass/support_data/body_models/dmpls/{}/model.npz".format(
#         subject_gender,
#     ),
# )
#
# num_betas = 16  # number of body parameters
# num_dmpls = 8  # number of DMPL parameters
#
# bm = BodyModel(
#     bm_fname=bm_fname,
#     num_betas=num_betas,
#     num_dmpls=num_dmpls,
#     dmpl_fname=dmpl_fname,
# ).to(comp_device)
#
#
# bdata_dict = dict(bdata)  # Convert npz object to a dictionary
# bdata_dict["poses"] = bdata_dict["poses"].copy()
#
# manifold = SpecialOrthogonal3(device=device, data_type=torch.float32)
#
# final_condition = bdata_dict["poses"][125, 3:66]
# final_condition = torch.tensor(final_condition, device=device, dtype=torch.float32)
# final_condition = final_condition.repeat(num_samples, 1)
# final_condition = manifold.exp(
#     torch.eye(3, device=device),
#     final_condition.view(num_samples, 21, 3),
# )
# all_joint_indices = [i for i in range(21)]
# # random_joint_indices = [i for i in range(21) if i not in [11, 12, 13, 14]]
# random_joint_indices = all_joint_indices
# fixed_joint_indices = np.setdiff1d(all_joint_indices, random_joint_indices)
#
# random_joints = (
#     sample_uniform_so3(num_samples * len(random_joint_indices)).to(device).unsqueeze(0)
# )
# random_joints = random_joints.view(num_samples, len(random_joint_indices), 3, 3)
#
#
# final_condition[:, random_joint_indices, :, :] = random_joints
#
# final_condition = final_condition.view(num_samples * 21, 3, 3)
#
# time_encoder = create_encoder(CONFIG)
#
#
# for j in range(num_times + 1):
#     encoded_time = time_encoder(time)
#
#     encoded_time = encoded_time.view(1, 2 * 32)
#     encoded_time_data = encoded_time.repeat(num_samples, 1)
#     final_condition_data = final_condition.view(num_samples, 21, 3, 3)
#     final_condition_data = final_condition.view(num_samples, 21, 9)
#     final_condition_data = final_condition.view(num_samples, 21 * 9)
#
#     outputs = model(final_condition_data, encoded_time_data)
#     outputs = outputs.view(num_samples, 21, 3)
#     outputs[:, fixed_joint_indices, :] = torch.zeros(
#         size=(num_samples, fixed_joint_indices.shape[0], 3),
#         device=device,
#         dtype=torch.float32,
#     )
#     outputs = outputs.view(num_samples, 63)
#     # print(outputs.shape)
#
#     final_condition = manifold.exp(
#         final_condition,
#         1.0 * dt * outputs.view(num_samples, 21, 3).view(num_samples * 21, 3),
#     )
#     time = time - dt
#
#     if j % 10 == 0:
#         bdata_dict["poses"][:num_samples, 3:66] = (
#             manifold.log(torch.eye(3, device=device), final_condition)
#             .view(num_samples, 21, 3)
#             .flatten(-2)
#             .cpu()
#             .detach()
#             .numpy()
#         )
#
#         bdata = bdata_dict
#
#         faces = c2c(bm.f)
#
#         # bdata["poses"][:num_samples] = np.copy(manifold.log(torch.eye(3, device=device), final_condition).view(num_samples, 21, 3).flatten(-2).cpu().detach().numpy())
#
#         time_length = len(bdata["trans"])
#
#         body_parms = {
#             "root_orient": torch.Tensor(bdata["poses"][:, :3]).to(
#                 comp_device,
#             ),  # controls the global root orientation
#             "pose_body": torch.Tensor(bdata["poses"][:, 3:66]).to(
#                 comp_device,
#             ),  # controls the body
#             "pose_hand": torch.Tensor(bdata["poses"][:, 66:]).to(
#                 comp_device,
#             ),  # controls the finger articulation
#             "trans": torch.Tensor(bdata["trans"]).to(
#                 comp_device,
#             ),  # controls the global body position
#             "betas": torch.Tensor(
#                 np.repeat(
#                     bdata["betas"][:num_betas][np.newaxis],
#                     repeats=time_length,
#                     axis=0,
#                 ),
#             ).to(comp_device),  # controls the body shape. Body shape is static
#             "dmpls": torch.Tensor(bdata["dmpls"][:, :num_dmpls]).to(
#                 comp_device,
#             ),  # controls soft tissue dynamics
#         }
#
#         body_pose_beta = bm(
#             **{k: v for k, v in body_parms.items() if k in ["pose_body", "betas"]},
#         )
#
#         vis_body_pose_beta(fId=0)
#         plt.savefig(f"./output_files/sampling/martial_arts_{(j + 0):04}.jpg")
#         plt.close()
