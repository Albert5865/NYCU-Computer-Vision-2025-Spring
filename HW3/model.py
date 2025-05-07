import torch.nn as nn


class CustomBackbone(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.stem = nn.Sequential(*list(base_model.children())[:4])
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.out_channels = [256, 512, 1024,2048] 

        def forward(self, x):
            x = self.stem(x)
            c2 = self.layer1(x)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            return {
                "layer1": c2,
                "layer2": c3,
                "layer3": c4,
                "layer4": c5,
            }
