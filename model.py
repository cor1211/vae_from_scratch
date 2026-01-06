import torch
import torch.nn as nn
import numpy as np
from torch.distributions.kl import kl_divergence

def get_norm_layer(dim:int = 2, norm: str='', out_channel: int = 0, num_group: int = 0):
    if norm.lower() == 'bn':
        if not out_channel:
            raise ValueError(f'Out channel of bn must not equal {out_channel}')
        else:
            return nn.BatchNorm2d(num_features=out_channel) if dim == 2 else nn.BatchNorm1d(num_features=out_channel)
    
    elif norm.lower() == 'gn':
        if not out_channel or (out_channel % num_group) != 0:
           raise ValueError(f'Out channel: {out_channel} cannot devided by group_channel {num_group}')
        else:
            return nn.GroupNorm(num_groups=num_group) 

    else:
        return nn.Identity()


def conv2d(in_conv, out_conv, kernel_size: int, relu: bool=False, max_pool: bool = False, norm: str = '') -> nn.Sequential:
    layer = nn.Sequential()
    layer.append(nn.Conv2d(in_channels=in_conv, out_channels=out_conv, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
    if relu:
        layer.append(nn.ReLU(True))
    if max_pool:
        layer.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
    if norm:
        layer.append(get_norm_layer(norm=norm, out_channel=out_conv))

    return layer


def linear(in_features, out_features, relu: bool = False, norm:str = ''):
    layer = nn.Sequential()
    layer.append(nn.Linear(in_features=in_features, out_features=out_features))
    if relu:
        layer.append(nn.ReLU(True))
    layer.append(get_norm_layer(dim = 1, norm=norm, out_channel=out_features))
    return layer


def transposed_conv(in_conv:int, out_conv:int, kernel_size:int, relu:bool = False, unpool: bool = False, norm:str = ''):
    layer = nn.Sequential()
    layer.append(nn.ConvTranspose2d(in_channels=in_conv, out_channels=out_conv, kernel_size=kernel_size, padding=kernel_size//2))
    if relu:
        layer.append(nn.ReLU(True))
    if unpool:
        layer.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
    if norm:
        layer.append(get_norm_layer(dim=2, norm=norm, out_channel=out_conv))
    return layer



class Encoder(nn.Module):
    def __init__(self, in_channels, dim_latent, device):
        super().__init__()
        self.device = device
        self.dim_latent = dim_latent
        # Layer 1
        self.layer_1 = conv2d(in_conv=in_channels, out_conv=8, kernel_size=5, relu=True, max_pool=True, norm='')
        self.bn_1 = get_norm_layer(dim=2, norm='bn', out_channel=8)
        # layer 2 
        self.layer_2 = conv2d(in_conv=8, out_conv=16, kernel_size=3, relu=True, max_pool=False, norm='')
        self.bn_2 = get_norm_layer(dim=2, norm='bn', out_channel=16)
        # layer 3
        self.layer_3 = conv2d(in_conv=16, out_conv=32, kernel_size=3, relu=True, max_pool=True, norm='')
        self.bn_3 = get_norm_layer(dim=2, norm='bn', out_channel=32)
        # layer 4
        self.layer_4 = conv2d(in_conv=32, out_conv=64, kernel_size=3, relu=True, max_pool=False, norm='')
        self.bn_4 = get_norm_layer(dim=2, norm='bn', out_channel=64)
        # layer 5
        self.layer_5 = linear(in_features=32*32*64, out_features=4096, relu=True, norm='')
        self.bn_5 = get_norm_layer(dim=1, norm='bn', out_channel=4096)
        # layer 6
        self.layer_6 = linear(in_features=4096, out_features=dim_latent * 2)
        # Make to squential
        self.encoder = nn.Sequential(self.layer_1, self.layer_2, self.layer_3, self.layer_4)
        self.latent = nn.Sequential(self.layer_5, self.layer_6)

    def forward(self, x):
        self.indices = []
        # x = self.encoder(x)
        x, indice= self.layer_1(x)
        x = self.bn_1(x)
        self.indices.append(indice)
        # print(indice.shape)
        x = self.layer_2(x)
        x = self.bn_2(x)
        
        x, indice = self.layer_3(x)
        x = self.bn_3(x)
        self.indices.append(indice)

        x = self.layer_4(x)
        x = self.bn_4(x)

        x = torch.flatten(x, start_dim=1)
        x = self.latent(x)
        x = x.view(-1, self.dim_latent, 2)
        # # print(x.shape)
        
        mean = x[:, :, 0]
        # print(f'Mean shape: {mean.shape}')
        logvar= x[:, :, 1]
        # print(f'LogVar shape: {logvar.shape}')
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z, self.indices


    def reparameterize(self, mean, logvar):
        # Convert logvar => std
        """
        Docstring for reparameterize
        Logvar = log(std^2) = 2log(std) => std = e^(1/2 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        # print(f'std shape: {std.shape}')
        # Reparameterization Trick: z = mean + std * epsilon
        epsilon = torch.rand_like(std).to(self.device)
        # print(f'epsilon shape: {epsilon.shape}')
        z = mean + std * epsilon
        # print(f'Z shape: {z.shape}')
        return z
    


class Decoder(nn.Module):
    def __init__(self, dim_latent, out_channels, indices):
        super().__init__()
        self.indices = indices
        
        self.layer_6d = linear(in_features= dim_latent, out_features=4096, relu=True, norm='bn')
        
        self.layer_5d = linear(in_features= 4096, out_features=4096, relu=True, norm='bn')
        
        self.layer_4d = transposed_conv(in_conv=4, out_conv=32, kernel_size=3, relu=True)
        self.unpool_4d = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.bn_4 = get_norm_layer(2, norm='bn', out_channel=32)

        self.layer_3d = transposed_conv(in_conv=32, out_conv=16, kernel_size=3, relu=True, norm='bn')

        self.layer_2d = transposed_conv(in_conv=16, out_conv=8, kernel_size=3, relu=True)
        self.unpool_2d = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.bn_2 = get_norm_layer(dim=2, norm='bn', out_channel=8)

        self.layer_1d = transposed_conv(in_conv=8, out_conv= out_channels, kernel_size=5, norm='bn')
        self.last_act = nn.Tanh()

    def forward(self, x):
        x = self.layer_6d(x)
        x = self.layer_5d(x)
        x = x.view(-1, 4, 32, 32)

        x = self.layer_4d(x)
        x = self.unpool_4d(x, self.indices[1])
        x = self.bn_4(x)
        
        x = self.layer_3d(x)

        x = self.layer_2d(x)
        x = self.unpool_2d(x, self.indices[0])
        x = self.bn_2(x)

        x = self.layer_1d(x)
        return self.last_act(x) # Range casting to [-1, 1]

class VAE(nn.Module):                                                                                             
    def __init__(self, in_channels, out_channels, dim_latent, device):
        super().__init__()
        self.indices = []
        self.encoder = Encoder(in_channels=in_channels, dim_latent=dim_latent, device = device)
        self.decoder = Decoder(dim_latent=dim_latent, out_channels=out_channels, indices= self.indices)

    def forward(self, x):
        mean, logvar, x, self.indices = self.encoder(x)
        self.decoder.indices = self.indices
        x = self.decoder(x)
        return mean, logvar, x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.rand(4, 3, 128, 128).to(device)

    vae = VAE(in_channels=3, out_channels=3, dim_latent=100, device = device).to(device)
    mean, var, out = vae(input)
    print(out.shape)


 