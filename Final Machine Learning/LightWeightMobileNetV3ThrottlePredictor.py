import torch.nn as nn
import timm

class LightWeightMobileNetV3ThrottlePredictor(nn.Module):
    def __init__(self):
        super(LightWeightMobileNetV3ThrottlePredictor, self).__init__()
        # pretrained MobileNetV3 model
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)

        # change to accommodate 4 inputs (RGB + Depth) and keep the rest
        self.model.conv_stem = nn.Conv2d(
            4,
            self.model.conv_stem.out_channels,
            kernel_size=self.model.conv_stem.kernel_size,
            stride=self.model.conv_stem.stride,
            padding=self.model.conv_stem.padding,
            bias=False
        )

        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Single output for throttle
        )

    def forward(self, x):
        return self.model(x)
