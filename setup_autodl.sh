#!/bin/bash
# AutoDL 4090 部署脚本: 设环境变量 + 验证 + 启动 CV
# 用法: source setup_autodl.sh  (注意是 source 不是 bash, 不然 env 不生效)

set -e

# ============ 1. 4090 优化参数 (覆盖 project_constants.py 默认) ============
export HANDCHAR_BATCH_SIZE=256       # 4090 24GB, 从 64 提到 256
export HANDCHAR_NUM_WORKERS=8        # Linux 多进程 dataloader
export HANDCHAR_LR=0.004             # 随 batch 4x 线性 scale (0.001 -> 0.004)

# 数据集路径 (上传到 /root/autodl-tmp/EnglishImg 后改这里)
export HANDCHAR_DATA_DIR="/root/autodl-tmp/EnglishImg/English"

# ============ 2. 环境检查 ============
echo "===== 环境检查 ====="
python -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available'
cc = torch.cuda.get_device_capability()
assert cc[0] >= 8, f'Need Ampere/Ada GPU, got SM_{cc[0]}{cc[1]}'
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)} (SM_{cc[0]}{cc[1]})')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
"

# ============ 3. 项目状态确认 ============
echo ""
echo "===== 数据集检查 ====="
if [ -d "$HANDCHAR_DATA_DIR/Img" ]; then
    echo "数据目录存在: $HANDCHAR_DATA_DIR"
    echo "图片数量: $(ls $HANDCHAR_DATA_DIR/Img | wc -l)"
else
    echo "ERROR: 数据集未找到, 请先上传到 $HANDCHAR_DATA_DIR"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "===== 当前配置 ====="
python -c "
from project_constants import BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, DEVICE
print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'NUM_WORKERS: {NUM_WORKERS}')
print(f'LEARNING_RATE: {LEARNING_RATE}')
print(f'DEVICE: {DEVICE}')
"

echo ""
echo "===== 部署就绪 ====="
echo "推荐启动命令 (按阶段):"
echo "  1. 阶段1 (架构 + 增强):   python compare.py --refined --epochs 50 --pairs"
echo "  2. 阶段2 (+GRN):          python compare.py --models 'refined_grn_*' --epochs 50 --pairs  # 待 all_models 注册"
echo "  3. RNN 位置消融:           python compare.py --refined --attach --epochs 50 --pairs"
