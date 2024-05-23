import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

weight_path = 'generator_epoch_50.pth'
num_images = 100
output_dir = 'inferenced_images/'

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image):
        if self.training:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            return image + self.weight * noise
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.noise1 = NoiseInjection(in_channels)
        self.noise2 = NoiseInjection(in_channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.noise1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.noise2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out = residual + 0.3 * out
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.noise = NoiseInjection(out_channels)

    def forward(self, x):
        out = self.upsample(x)
        residual = self.adjust_channels(out)
        out = self.conv(out)
        out = self.noise(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = residual + 0.3 * out
        return out

class ResNetGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, dropout_rate=0.5):
        super(ResNetGenerator, self).__init__()
        
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512*self.init_size*self.init_size))
        self.noise_injection1 = NoiseInjection(512)
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64)
        )
        
        self.upsampling_blocks = nn.Sequential(
            UpsampleBlock(512, 256, dropout_rate),
            UpsampleBlock(256, 128, dropout_rate),
            UpsampleBlock(128, 64, dropout_rate)
        )
        
        self.final_conv = nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        out = self.upsampling_blocks(out)
        out = self.residual_blocks(out)
        out = self.final_conv(out)
        out = self.tanh(out)
        return out

latent_dim = 100
img_channels = 3

weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weight_path)

generator = ResNetGenerator(latent_dim, img_channels)
generator.load_state_dict(torch.load(weight_path))

generator.eval()

z = torch.randn(num_images, latent_dim)
with torch.no_grad():
    fake_images = generator(z)

fake_images = (fake_images + 1) / 2
fake_images = fake_images.permute(0, 2, 3, 1).cpu().numpy()

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(num_images):
    plt.imsave(os.path.join(output_dir, f'generated_image_{i}.png'), fake_images[i])
    
print(f'{num_images} images are generated and saved in {output_dir}')
