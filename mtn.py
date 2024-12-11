import torch.nn as nn
import torch

from mmdet.registry import MODELS
from mmdet.utils import ConfigType


@MODELS.register_module()
class MTN(nn.Module):
    def __init__(self, encoder: ConfigType, decoder: ConfigType):
        super(MTN, self).__init__()
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



@MODELS.register_module()
class FCEncoder(nn.Module):
    def __init__(self, time_step):
        super(FCEncoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(time_step * 270, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # 最终输出128维
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

 
@MODELS.register_module()
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        
        # 定义一系列一维卷积层
        self.conv1 = nn.Conv1d(in_channels=270, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)

        
        # 展平和全连接层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8, 128)  # 32 是最后一层的通道数，2 是时间步的长度
    
    def forward(self, x):
        # 输入 x 的 shape 是 (batch_size, 30, 270)，需要转换为 (batch_size, 270, 30)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        
        # 通过卷积层
        x = self.conv1(x)  # 输出维度: (batch_size, 256, 15)
        x = torch.relu(x)
        
        x = self.conv2(x)  # 输出维度: (batch_size, 128, 8)
        x = torch.relu(x)
        
        # 展平并传入全连接层
        x = self.flatten(x)  # 输出维度: (batch_size, 128 * 8)
        x = self.fc(x)       # 输出维度: (batch_size, 128)
        
        return x
    
@MODELS.register_module()
class ConvEncoderLargeBottleneck(nn.Module):
    def __init__(self):
        super(ConvEncoderLargeBottleneck, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=270, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(8)

        self.linear = nn.Sequential(
            nn.Linear(8 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 2160),
        )
        
    
    def forward(self, x):
        # 输入 x 的 shape 是 (batch_size, 30, 270)，需要转换为 (batch_size, 270, 30)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)

        # z-score
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-5)
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        
        return x

    
@MODELS.register_module()
class ImageDecoderLargeBottleneck(nn.Module):
    def __init__(self):
        super(ImageDecoderLargeBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=2, padding=1)
        

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1, padding=1)

        self.conv8 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):

        
        # 调整形状为 (batch_size, 1, 36, 60)
        x = x.view(-1, 1, 36, 60)

        x = self.conv1(x) # 调整形状为 (batch_size, 32, 18, 30)
        x = self.relu(x)

        x = self.conv2(x) # 调整形状为 (batch_size, 64, 9, 15)
        x = self.relu(x)
        
        x = self.upsample(x) # 调整形状为 (batch_size, 64, 18, 30)
        x = self.conv3(x)

        x = self.upsample(x) # 调整形状为 (batch_size, 32, 36, 60)
        x = self.conv4(x)

        x = self.upsample(x) # 调整形状为 (batch_size, 16, 72, 120)
        x = self.conv5(x)

        x = self.upsample(x) # 调整形状为 (batch_size, 8, 144, 240)
        x = self.conv6(x)

        x = self.upsample(x) # 调整形状为 (batch_size, 4, 288, 480)
        x = self.conv7(x)

        x = self.upsample(x) # 调整形状为 (batch_size, 2, 576, 960)

        x = self.conv8(x)
        x = self.relu(x)
        
        return x
    
@MODELS.register_module()
class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=2, padding=1)
        
        # 逐步上采样的卷积转置层，每层放大 4 倍
        self.deconv1 = nn.ConvTranspose2d(192, 64, kernel_size=4, stride=4, padding=0)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=4, padding=0)
        
        # 激活函数
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = x.view(-1, 1, 36, 60)

        x = self.conv1(x) # 调整形状为 (batch_size, 64, 18, 30)
        x = self.relu(x)

        x = self.conv2(x) # 调整形状为 (batch_size, 192, 9, 15)
        x = self.relu(x)

        # 逐步通过卷积转置层上采样，每层放大 4 倍
        x = self.deconv1(x)  # 输出维度: (batch_size, 64, 32, 60)
        x = self.relu(x)

        
        x = self.deconv2(x)  # 输出维度: (batch_size, 32, 128, 240)
        x = self.relu(x)

        
        x = self.deconv3(x)  # 输出维度: (batch_size, 3, 576, 960)
        x = self.relu(x)
        
        return x
    
if __name__ == '__main__':
    model = ConvEncoderLargeBottleneck()

    x = torch.rand(16, 32, 3, 3, 30)
    print(model(x).shape)
