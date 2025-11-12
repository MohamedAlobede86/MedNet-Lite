import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D-adapted Convolutional Block Attention Module (CBAM)
class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Temporal Attention
        self.conv_temporal = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_temporal = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        b, c, t = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        ca = self.sigmoid_channel(avg_out + max_out).view(b, c, 1)
        x = x * ca  # Apply channel attention

        # Temporal Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        ta_input = torch.cat([avg_pool, max_pool], dim=1)
        ta = self.sigmoid_temporal(self.conv_temporal(ta_input))
        x = x * ta  # Apply temporal attention

        return x, ca, ta  # Return attention maps for interpretability

# MedNet-Lite Architecture
class MedNetLite(nn.Module):
    def __init__(self):
        super(MedNetLite, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.cbam1 = CBAM1D(16)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.cbam2 = CBAM1D(32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: [batch_size, 1, 300]
        x = self.conv_block1(x)
        x, ca1, ta1 = self.cbam1(x)

        x = self.conv_block2(x)
        x, ca2, ta2 = self.cbam2(x)

        x = self.pool(x).view(x.size(0), -1)
        logit = self.fc(x)
        output = self.sigmoid(logit)

        # Return prediction and attention maps
        attention_maps = {
            'CA1': ca1,
            'TA1': ta1,
            'CA2': ca2,
            'TA2': ta2
        }
        return output, attention_maps
