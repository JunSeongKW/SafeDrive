#!/bin/bash
# SafeDrive 전용 conda env 구축 (기존 env 는 건드리지 않음)
set -e
source /home/kaist5/miniconda3/etc/profile.d/conda.sh
cd /home/kaist5/data/junseong/SafeDrive

ENV=safedrive
echo "=== [1/6] conda env 생성 (python 3.10) ==="
conda create -n $ENV python=3.10 -y

echo "=== [2/6] torch 2.1.0 + cu118 ==="
conda run -n $ENV --no-capture-output pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== [3/6] navsim requirements (nuplan-devkit 포함) ==="
conda run -n $ENV --no-capture-output pip install --no-cache-dir -r requirements.txt

echo "=== [4/6] navsim 패키지 설치 (editable) ==="
conda run -n $ENV --no-capture-output pip install --no-cache-dir -e .

echo "=== [5/6] mmcv 2.1.0 ==="
conda run -n $ENV --no-capture-output pip install --no-cache-dir openmim
conda run -n $ENV --no-capture-output mim install mmcv==2.1.0

echo "=== [6/6] spconv + 기타 ==="
conda run -n $ENV --no-capture-output pip install --no-cache-dir einops lmdb spconv-cu118

echo "=== 검증 ==="
conda run -n $ENV --no-capture-output python -c "
import torch, mmcv, mmengine, spconv, timm, numpy
print('torch   ', torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('mmcv    ', mmcv.__version__)
print('mmengine', mmengine.__version__)
print('spconv  ', spconv.__version__ if hasattr(spconv,'__version__') else 'ok')
print('numpy   ', numpy.__version__)
import mmcv.ops  # ABI 체크
print('mmcv._ext OK')
"
echo "BUILD_DONE"
