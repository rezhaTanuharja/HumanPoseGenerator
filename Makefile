ENV_NAME=pose_generator

PYTHON_VERSION=3.8.5
PYTORCH_VERSION=2.0.1
CUDATOOLKIT_VERSION=11.8
TORCHVISION_VERSION=0.15.0

.PHONY: all
all: install

.PHONY: dev-setup
dev-setup: install
	conda run --name $(ENV_NAME) pip install pyright debugpy ruff pytest

.PHONY: install
install: create_env install_amass install_manifold_diffusion

.PHONY: create_env
create_env:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) pytorch=$(PYTORCH_VERSION) cudatoolkit=$(CUDATOOLKIT_VERSION) torchvision -c pytorch -y

.PHONY: install_amass
install_amass:
	conda run --name $(ENV_NAME) pip install git+https://github.com/nghorbani/human_body_prior.git
	conda run --name $(ENV_NAME) pip install pyrender==0.1.45
	conda run --name $(ENV_NAME) pip install git+https://github.com/nghorbani/body_visualizer.git

.PHONY: install_manifold_diffusion
install_manifold_diffusion:
	conda run --name $(ENV_NAME) pip install git+https://github.com/rezhaTanuharja/manifoldsDiffusion.git

.PHONY: clean
clean:
	conda env remove --name $(ENV_NAME) -y
