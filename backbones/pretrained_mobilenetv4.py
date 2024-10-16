import torch
import torch.nn as nn
import timm


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Global Average Pooling to make it compatible with the fully connected layer
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3], keepdim=False)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y  # Apply channel attention


class CNN(nn.Module):
    def __init__(self, n_classes=6):
        super(CNN, self).__init__()
        # Use MobileNetV4 as the base model
        self.base_model = timm.create_model('mobilenetv4_conv_large', pretrained=True)

        in_features = self.base_model.num_features
        self.base_model.classifier = nn.Identity()
        
        self.se_module = SEBlock(in_features)
        self.self_attention = SelfAttention(in_features)  # Add self-attention layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classification main task
        self.fc_main = nn.Linear(in_features, n_classes)
        
        # supplementary tasks
        self.fc_shapeset = nn.Linear(in_features, 2)
        self.fc_type = nn.Linear(in_features, 2)
        self.fc_total_height = nn.Linear(in_features, n_classes)  # classification
        self.fc_instability = nn.Linear(in_features, 3)
        self.fc_cam_angle = nn.Linear(in_features, 2)

        # learnable log sig
        self.log_sigma_main = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_shapeset = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_type = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_total_height = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_instability = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cam_angle = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        x = self.base_model.forward_features(x)

        # print(x.shape)
        
        # attention
        # Apply Self-Attention (preserving spatial dimensions)
        x = self.self_attention(x)
        # Apply SEBlock for channel attention (reduce spatial dimensions)
        x = self.se_module(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        out_main = self.fc_main(x)
        
        out_shapeset = self.fc_shapeset(x)
        out_type = self.fc_type(x)
        out_total_height = self.fc_total_height(x)
        out_instability = self.fc_instability(x)
        out_cam_angle = self.fc_cam_angle(x)
        
        return out_main, out_shapeset, out_type, out_total_height, out_instability, out_cam_angle
