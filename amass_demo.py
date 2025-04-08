# import torch.cuda
from os import path as osp

import matplotlib.pyplot as plt

# import torch.cuda
import numpy as np
import torch
import torch.nn.utils.prune
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors, show_image
from diffusionmodels.manifolds.rotationalgroups import SpecialOrthogonal3
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# from body_visualizer.mesh.sphere import points_to_spheres
from models.modulators import FiLM
from pipelines.encoders import time_encoder


def sample_uniform_so3(n=1):
    """Generate n random rotations uniformly sampled from SO(3) using quaternions."""
    u1, u2, u3 = torch.rand(3, n)  # Sample three independent uniform variables

    # Compute quaternion components
    q0 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2)
    q1 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2)
    q2 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
    q3 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

    # Form rotation matrices
    R = torch.stack(
        [
            1 - 2 * (q2**2 + q3**2),
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
            2 * (q1 * q2 + q0 * q3),
            1 - 2 * (q1**2 + q3**2),
            2 * (q2 * q3 - q0 * q1),
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            1 - 2 * (q1**2 + q2**2),
        ],
        dim=-1,
    ).reshape(n, 3, 3)

    return R


num_times = 1000
dt = 1.6 / num_times

device = torch.device("cuda")

activation_layer = torch.nn.LeakyReLU(negative_slope=0.005)
dropout_layer = torch.nn.Dropout(p=0.0)

temporal_components = torch.nn.Sequential(
    torch.nn.Linear(2 * 32, 32),
    activation_layer,
    torch.nn.Linear(32, 32),
    activation_layer,
    dropout_layer,
    torch.nn.Linear(32, 52 * 18),
    activation_layer,
).to(device)

spatial_components = torch.nn.Sequential(
    torch.nn.Linear(52 * 9, 100),
    activation_layer,
    dropout_layer,
    torch.nn.Linear(100, 100),
    activation_layer,
    torch.nn.Linear(100, 52 * 3),
).to(device)

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


model = FiLM(modulator=temporal_components, head=spatial_components).to(device)

checkpoint = torch.load("checkpoints/human.pth", weights_only=True)

state_dict = checkpoint["model_state_dict"]
module_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# model.load_state_dict(checkpoint["model_state_dict"])
model.load_state_dict(module_state_dict)
model.eval()

time = torch.tensor(
    [
        1.6,
    ],
    dtype=torch.float32,
    device=device,
)

final_condition = sample_uniform_so3(20 * 52).to(device).unsqueeze(0)

manifold = SpecialOrthogonal3(device=device, data_type=torch.float32)

for j in range(num_times + 1):
    encoded_time = time_encoder(time, 1.6, num_waves=32)

    encoded_time = encoded_time.view(1, 2 * 32)
    encoded_time_data = encoded_time.repeat(20, 1)
    final_condition_data = final_condition.view(20, 52, 3, 3)
    final_condition_data = final_condition.view(20, 52, 9)
    final_condition_data = final_condition.view(20, 52 * 9)

    outputs = model(final_condition_data, encoded_time_data)

    final_condition = manifold.exp(
        final_condition, -1.0 * dt * outputs.view(20, 52, 3).view(20 * 52, 3)
    )
    time = time - dt


support_dir = "/home/ratanuh/Datasets/"

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# comp_device = torch.device("cpu")

amass_npz_fname = osp.join(
    support_dir, "ACCAD/Male2Running_c3d/C18 - run to hop to walk_poses.npz"
)  # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = "male"

print("Data keys available:%s" % list(bdata.keys()))

print("The subject of the mocap sequence is  {}.".format(subject_gender))

bm_fname = osp.join(
    support_dir,
    "/home/ratanuh/Projects/HumanFilter/amass/support_data/body_models/smplh/{}/model.npz".format(
        subject_gender
    ),
)
dmpl_fname = osp.join(
    support_dir,
    "/home/ratanuh/Projects/HumanFilter/amass/support_data/body_models/dmpls/{}/model.npz".format(
        subject_gender
    ),
)

num_betas = 16  # number of body parameters
num_dmpls = 8  # number of DMPL parameters

bm = BodyModel(
    bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname
).to(comp_device)


bdata_dict = dict(bdata)  # Convert npz object to a dictionary
bdata_dict["poses"] = bdata_dict["poses"].copy()

bdata_dict["poses"][:20] = (
    manifold.log(torch.eye(3, device=device), final_condition)
    .view(20, 52, 3)
    .flatten(-2)
    .cpu()
    .detach()
    .numpy()
)

bdata = bdata_dict

faces = c2c(bm.f)


# bdata["poses"][:20] = np.copy(manifold.log(torch.eye(3, device=device), final_condition).view(20, 52, 3).flatten(-2).cpu().detach().numpy())

time_length = len(bdata["trans"])

body_parms = {
    "root_orient": torch.Tensor(bdata["poses"][:, :3]).to(
        comp_device
    ),  # controls the global root orientation
    "pose_body": torch.Tensor(bdata["poses"][:, 3:66]).to(
        comp_device
    ),  # controls the body
    "pose_hand": torch.Tensor(bdata["poses"][:, 66:]).to(
        comp_device
    ),  # controls the finger articulation
    "trans": torch.Tensor(bdata["trans"]).to(
        comp_device
    ),  # controls the global body position
    "betas": torch.Tensor(
        np.repeat(bdata["betas"][:num_betas][np.newaxis], repeats=time_length, axis=0)
    ).to(comp_device),  # controls the body shape. Body shape is static
    "dmpls": torch.Tensor(bdata["dmpls"][:, :num_dmpls]).to(
        comp_device
    ),  # controls soft tissue dynamics
}

print(
    "Body parameter vector shapes: \n{}".format(
        " \n".join(["{}: {}".format(k, v.shape) for k, v in body_parms.items()])
    )
)
print("time_length = {}".format(time_length))


imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)


body_pose_beta = bm(
    **{k: v for k, v in body_parms.items() if k in ["pose_body", "betas"]}
)


def vis_body_pose_beta(fId=0):
    body_mesh = trimesh.Trimesh(
        vertices=c2c(body_pose_beta.v[fId]),
        faces=faces,
        vertex_colors=np.tile(colors["grey"], (6890, 1)),
    )
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)


for i in range(0, 20):
    vis_body_pose_beta(fId=i)
    plt.savefig(f"./output_files/ground_truth/running_{i:03}.jpg")
    plt.close()
