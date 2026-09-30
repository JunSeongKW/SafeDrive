#!/bin/bash
set -e
source /home/kaist5/miniconda3/etc/profile.d/conda.sh
conda activate safedrive
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"      # H100 전용으로만 빌드 -> 컴파일 시간 단축
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=32                      # 공유 머신: 192코어 중 32개만
echo "nvcc: $(nvcc --version | tail -2 | head -1)"
pip uninstall -y mmcv 2>/dev/null || true
pip install --no-cache-dir --no-build-isolation --no-binary mmcv mmcv==2.1.0
echo "MMCV_BUILD_DONE"
