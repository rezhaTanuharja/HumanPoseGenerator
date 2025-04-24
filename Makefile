ENV_NAME=pose_generator

PYTHON_VERSION=3.8.5
MATPLOTLIB_VERSION=3.7.5
PYRENDER_VERSION=0.1.45
OPENCV_VERSION=4.11.0.86

.PHONY: all
all: install

.PHONY: dev-setup
dev-setup: install
	conda run --name $(ENV_NAME) pip install pyright debugpy ruff pytest

.PHONY: install
install: create_env install_amass install_manifold_diffusion

.PHONY: create_env
create_env:
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -c conda-forge -y
	conda run --name $(ENV_NAME) pip install pyrender==$(PYRENDER_VERSION) matplotlib==$(MATPLOTLIB_VERSION) opencv-python==$(OPENCV_VERSION)

.PHONY: install_manifold_diffusion
install_manifold_diffusion:
	conda run --name $(ENV_NAME) pip install git+https://github.com/rezhaTanuharja/manifoldsDiffusion.git

.PHONY: install_amass
install_amass:
	conda run --name $(ENV_NAME) pip install git+https://github.com/nghorbani/human_body_prior.git
	conda run --name $(ENV_NAME) pip install git+https://github.com/nghorbani/body_visualizer.git

.PHONY: clean
clean:
	conda env remove --name $(ENV_NAME) -y

.PHONY: sync
sync:
	rsync -avzh --exclude=".git" --exclude-from=".gitignore" ./ ips:Projects/HumanPoseGenerator/

.PHONY: fetch
fetch:
	rsync -avzh --exclude=".git" --exclude-from=".gitignore" ips:Projects/HumanPoseGenerator/ ./
