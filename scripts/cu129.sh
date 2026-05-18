#!/bin/bash
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

unset https_proxy
rm -rf ~/.conda/pkgs/locks
rm -rf ~/mamba/pkgs/locks

mamba clean --all --yes
conda clean --all --yes

conda activate base
mamba env remove -n cu129 -y
conda create -n cu129 --clone myenv -y
pip config set global.index-url https://mirrors.xjtu.edu.cn/pypi/simple
mamba run -n cu129 pip install --root-user-action ignore torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
mamba run -n cu129 pip install --root-user-action ignore lightning==2.5.5 torchmetrics==1.8.2 wandb==0.22.0 pydantic==2.09.0
mamba run -n cu129 pip install --root-user-action ignore torch_geometric==2.7.0

# Optional dependencies:
mamba run -n cu129 pip install --root-user-action ignore pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
