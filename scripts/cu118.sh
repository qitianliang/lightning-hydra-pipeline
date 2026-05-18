#!/bin/bash
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

unset https_proxy
rm -rf ~/.conda/pkgs/locks
rm -rf ~/mamba/pkgs/locks

mamba clean --all --yes
conda clean --all --yes

conda activate base
mamba env remove -n cu118 -y
conda create -n cu118 --clone myenv
pip config set global.index-url https://mirrors.xjtu.edu.cn/pypi/simple
mamba run -n cu118 pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --find-links env/torch-2.1.1-cu118-py3.10 --index-url https://mirrors.xjtu.edu.cn/pypi/simple --root-user-action ignore
mamba run -n cu118 pip install --root-user-action ignore torch-geometric==2.4.0 einops==0.8.1 lightning==2.5.5 torchmetrics==1.8.2 numpy==1.26.4
mamba run -n cu118 pip install --root-user-action ignore pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.1%2Bcu118.html
