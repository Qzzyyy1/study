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
        
        # ==========================================
        # 1. 主路：原版空间 ResNet-Transformer (一字不改)
        # ==========================================
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1) 
        self.layer3 = self._make_layer(256, 2, stride=2) 

        final_h = (patch_size + 1) // 2 
        self.seq_len = final_h * final_h 
        embed_dim = 256
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=512, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim * self.seq_len, feature_dim)
        
        # ==========================================
        # 2. 辅路：纯净光谱分支 (Spectral Branch)
        # ==========================================
        # 仅使用 1x1 卷积处理中心像素，保护物理光谱曲线
        self.spectral_branch = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # ==========================================
        # 3. 核心：零初始化残差门 (Zero-Initialized Gate)
        # ==========================================
        # 初始化为 0。确保网络刚开始时完全等于你的 0.853 版本！
        self.spectral_gate = nn.Parameter(torch.zeros(1))
        
        self.pixel_mapper = nn.Conv2d(256, feature_dim, kernel_size=1)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # --- A. 运行主路 (Spatial-Transformer Flow) ---
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        pixel_feat = out 
        
        out = self.layer3(out) 
        out = out.flatten(2).transpose(1, 2) 
        out = self.transformer(out)
        out = out.flatten(1)
        spatial_features = self.fc(out) # (Batch, 288)

        # --- B. 运行辅路 (Pure Spectral Flow) ---
        # 提取中心目标像素 (因为 HSI 标注是基于中心像素的)
        center_y, center_x = h // 2, w // 2
        center_pixel = x[:, :, center_y:center_y+1, center_x:center_x+1] # (Batch, Channels, 1, 1)
        
        spectral_features = self.spectral_branch(center_pixel) # (Batch, 288, 1, 1)
        spectral_features = spectral_features.view(b, -1) # (Batch, 288)

        # --- C. 零初始化残差融合 (Safest Fusion) ---
        # 因为 self.spectral_gate 初始化为 0，所以初始状态 features = spatial_features
        # 随着训练，优化器会自动调整 gate 的大小，吸纳对分类有用的光谱信息
        features = spatial_features + self.spectral_gate * spectral_features

        # --- D. 球面保护 ---
        features = F.normalize(features, p=2, dim=1) * 16.0
        
        return {
            'pixel': pixel_feat, 
            'features': features
        }