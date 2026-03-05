import torch
import torch.nn as nn


class FastMLP(nn.Module):
    """
    超快速MLP图像分类器 - 专为32×32图像优化

    设计思路：
    1. 渐进式降维：3072 → 1024 → 512 → 256 → 128
    2. 使用ReLU激活（速度最快）
    3. BatchNorm加速收敛
    4. Dropout防止过拟合
    """

    def __init__(self, num_classes=10):
        super(FastMLP, self).__init__()

        # 输入：32×32×3 = 3072维
        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


if __name__ == "__main__":
    # 标准版（精度优先）
    model = FastMLP(num_classes=9)

    # 极速版（速度优先）
    # model = UltraFastMLP(num_classes=9)

    # 测试
    x = torch.randn(32, 3, 32, 32)  # batch_size=32
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 推理速度测试
    import time

    model.eval()
    test_cnt = 10000
    with torch.no_grad():
        start = time.time()
        for _ in range(test_cnt):
            _ = model(x)
        end = time.time()
    print(f"{test_cnt}次推理耗时: {(end - start) * 1000:.2f}ms")
    print(f"平均单次推理: {(end - start) * 1000 / test_cnt:.2f}ms")