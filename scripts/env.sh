#!/bin/bash
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

unset https_proxy
rm -rf ~/.conda/pkgs/locks
rm -rf ~/mamba/pkgs/locks

mamba clean --all --yes
conda clean --all --yes

conda activate base
# 删除所有锁文件
mamba env remove -n myenv -y
mamba env create -f environment.yaml -y
