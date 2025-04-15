# HumanPoseGenerator

Generate realistic human poses using manifold-aware conditional flow matching.

## Background

Conditional Flow Matching is a promising alternative to diffusion models for image generation, offering fast sampling and stable training.
While sample fidelity is still a challenge in image generation domain, such requirement is more relaxed in the domain of human pose generation.
Therefore, this project attempts to use conditional flow matching for human pose generation, aiming to accelerate the sampling process such that the generated human pose can be used by real-time downstream processes, e.g., human-robot interaction, autonomous driving, etc..
