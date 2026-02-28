import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, feature_dim=288):
        super(ResNetTransformer, self).__init__()
        
        # --- 1. 浅层 ResNet (不进行下采样) ---
        # 针对 7x7 输入，我们保持尺寸不变，只增加通道数
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 堆叠残差块
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1) 
        # 这里可以选择 stride=2 变成 4x4，或者保持 7x7。为了保留局部细节，建议只在最后一层下采样或不采样。
        # 这里我们做一次下采样：7x7 -> 4x4
        self.layer3 = self._make_layer(256, 2, stride=2) 

        # --- 2. Transformer Encoder ---
        # 计算序列长度: 如果 7x7 下采样一次变成 4x4 -> 16
        final_h = (patch_size + 1) // 2 
        self.seq_len = final_h * final_h 
        embed_dim = 256
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512, dropout=0.3, batch_first=True)
        # 注意：dropout 设为 0.3 以防止对源域过拟合
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- 3. 输出映射 ---
        self.fc = nn.Linear(embed_dim * self.seq_len, feature_dim)
        
        # 这里的 pixel_mapper 是为了兼容 WGDT 的输出格式
        self.pixel_mapper = nn.Conv2d(256, feature_dim, kernel_size=1)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Bands, H, W) -> 需要在 WGDT.py 里保证传入的是这个
        
        # 1. ResNet 特征提取
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        pixel_feat = out # (Batch, 128, 7, 7) - 保留中间特征给 'pixel'
        
        out = self.layer3(out) # (Batch, 256, 4, 4)
        
        # 2. Transformer
        b, c, h, w = out.shape
        out = out.flatten(2).transpose(1, 2) # (Batch, 16, 256)
        out = self.transformer(out)
        
        # 3. 展平并映射
        out = out.flatten(1)
        features = self.fc(out) # (Batch, 288)

        # === 关键修改：特征归一化 (保护未知类) ===
        # 将特征投影到单位球面上，防止 Transformer 输出模长过大的特征
        features = F.normalize(features, p=2, dim=1) * 16.0
        # *10.0 是为了给 softmax 提供足够的 logits 范围，类似 temperature scaling
        
        # 构造 pixel 特征 (简单调整通道数以匹配)
        # 注意：这里我们简单地返回 layer2 的特征，或者你可以用 layer3 插值回去
        # WGDT 似乎只用 pixel 做辅助，在此架构中 features 才是核心
        
        return {
            'pixel': pixel_feat, 
            'features': features
        }