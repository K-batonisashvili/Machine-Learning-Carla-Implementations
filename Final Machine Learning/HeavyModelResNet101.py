import torch.nn as nn
import timm

class HeavyModelResNet101(nn.Module):
    def __init__(self):
        super(HeavyModelResNet101, self).__init__()
        # pretrained resnet101 model
        self.model = timm.create_model('resnet101', pretrained=True)

        # change to accommodate 4 inputs (RGB + Depth) and keep the rest
        self.model.conv1 = nn.Conv2d(
            4,  # 4 input channels (RGB + Depth)
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False
        )

        # single throttle value output
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Single output for throttle
        )

    def forward(self, x):
        return self.model(x)