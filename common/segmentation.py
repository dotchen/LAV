from torch import nn

class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_labels):
        super().__init__()
        
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(input_channels,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,num_labels,1,1,0),
        )
        
    def forward(self, x):
        return self.upconv(x)
