import torch
import torch.nn as nn
# import pretrainedmodels
import timm


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
        y = self.fc(x)
        return x * y

class CNN(nn.Module):
    def __init__(self, n_classes=6):
        super(CNN, self).__init__()
        self.base_model = timm.create_model('inception_v4', pretrained=True)
        # self.base_model = pretrainedmodels.__dict__['inceptionv4'](pretrained='imagenet')

        # edge detection added
        self.base_model.features[0].conv = nn.Conv2d(4, 32, kernel_size=3, stride=2, bias=False)
        nn.init.kaiming_normal_(self.base_model.features[0].conv.weight, mode='fan_out', nonlinearity='relu')
        
        in_features = self.base_model.last_linear.in_features
        self.base_model.last_linear = nn.Identity()
        
        self.se_module = SEBlock(in_features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # regression main task
        # self.fc_main = nn.Linear(in_features, 1)

        # classification main task
        self.fc_main = nn.Linear(in_features, n_classes)
        
        # supplementary tasks
        self.fc_shapeset = nn.Linear(in_features, 2)
        self.fc_type = nn.Linear(in_features, 2)
        # self.fc_total_height = nn.Linear(in_features, n_classes)  # classification
        # self.fc_total_height = nn.Linear(in_features, 1)  # regression
        self.fc_num_unstable = nn.Linear(in_features, n_classes)  # 0-5 difference
        self.fc_instability = nn.Linear(in_features, 3)
        self.fc_cam_angle = nn.Linear(in_features, 2)

        # learnable log sig
        self.log_sigma_main = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_shapeset = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_type = nn.Parameter(torch.tensor(0.0))
        # self.log_sigma_total_height = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_num_unstable = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_instability = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cam_angle = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        x = self.base_model.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.se_module(x)
        
        out_main = self.fc_main(x)
        
        out_shapeset = self.fc_shapeset(x)
        out_type = self.fc_type(x)
        # out_total_height = self.fc_total_height(x)
        out_num_unstable = self.fc_num_unstable(x)
        out_instability = self.fc_instability(x)
        out_cam_angle = self.fc_cam_angle(x)
        
        return out_main, out_shapeset, out_type, out_num_unstable, out_instability, out_cam_angle

