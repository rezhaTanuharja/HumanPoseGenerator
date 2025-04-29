import torch
import torch.nn.utils.prune
from diffusionmodels import manifolds

from humanposegenerator import config, models, pipelines, time_integrators, utilities


def main():
    parameters = config.CONFIG["pose_generator"]
    parameters["batch_size"] = 1

    dataset = utilities.amass.load_and_combine_poses(
        parameters["data_directory"],
        parameters["joint_indices"],
    )

    num_big_samples = 400

    sample_indices = torch.randperm(n=parameters["total_size"])[:num_big_samples]

    dataset = dataset[sample_indices]

    num_small_samples = 100
    random_indices = torch.randperm(len(dataset))
    num_time_steps = 1200

    chosen_data = torch.tensor(
        dataset[random_indices[:num_big_samples]],
        device=parameters["device"],
        dtype=parameters["data_type"],
    )

    manifold = manifolds.rotationalgroups.SpecialOrthogonal3(
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    pose_samples = manifold.exp(
        torch.eye(3, device=parameters["device"], dtype=parameters["data_type"]),
        chosen_data,
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

    utilities.torch_module.set_dropout_rate(
        pose_generator,
        0.0,
    )
    encoder = pipelines.encoders.create_encoder(parameters)

    pose = utilities.sample.sample_uniform_so3(
        num_samples=num_small_samples * len(parameters["joint_indices"]) // 3,
    ).to(parameters["device"])

    pose = pose.view(num_small_samples, len(parameters["joint_indices"]) // 3, 3, 3)

    solver = time_integrators.explicit_euler.ExplicitEuler(
        velocity_model=pose_generator,
        time_encoder=encoder,
        manifold=manifold,
        device=parameters["device"],
        data_type=parameters["data_type"],
    )

    pose = solver.solve(
        initial_condition=pose,
        initial_time=parameters["period"],
        num_time_steps=num_time_steps,
    )

    pose = pose.unsqueeze(1).repeat(1, num_big_samples, 1, 1, 1)

    vectors = manifold.log(pose, pose_samples)
    geodesics = torch.sqrt(torch.norm(vectors, dim=-1))
    distance = torch.sum(geodesics, dim=-1)
    min_distance, _ = torch.min(distance, dim=-1)

    print(torch.mean(min_distance))
    print(torch.median(min_distance))
    print(torch.var(min_distance))


if __name__ == "__main__":
    main()
