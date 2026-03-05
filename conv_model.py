import torch
import torch.nn as nn


class ConvMLP(nn.Module):
    """
    卷积+MLP混合分类器 - 专为32×32灰度图像优化

    结构:
    1. 两层卷积提取特征
    2. 两层全连接分类
    """

    def __init__(self, num_classes=6, in_channels=1):
        super(ConvMLP, self).__init__()

        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积: 1 -> 32, 32x32 -> 16x16
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 第二层卷积: 32 -> 64, 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 展平层
        self.flatten = nn.Flatten()

        # 全连接层: 64*8*8 = 4096 -> 256 -> num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


class LightConvMLP(nn.Module):
    """
    轻量级版本 - 只用1层卷积
    """

    def __init__(self, num_classes=6, in_channels=1):
        super(LightConvMLP, self).__init__()

        # 单层卷积: 1 -> 64, 32x32 -> 16x16
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()

        # 全连接层: 64*16*16 = 16384 -> 512 -> num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


if __name__ == "__main__":
    # 测试两层卷积版本
    model = ConvMLP(num_classes=6, in_channels=1)

    # 测试单层卷积版本
    # model = LightConvMLP(num_classes=6, in_channels=1)

    # 测试输入 (batch_size=32, channels=1, height=32, width=32)
    x = torch.randn(32, 1, 32, 32)
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    # 推理速度测试
    import time

    model.eval()
    test_cnt = 1000
    with torch.no_grad():
        start = time.time()
        for _ in range(test_cnt):
            _ = model(x)
        end = time.time()
    print(f"{test_cnt}次推理耗时: {(end - start) * 1000:.2f}ms")
    print(f"平均单次推理: {(end - start) * 1000 / test_cnt:.2f}ms")