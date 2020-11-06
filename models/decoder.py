import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class SpatialBroadcastDec(nn.Module):
    """Variational Autoencoder with spatial broadcast decoder"""
    def __init__(self, input_dim, output_dim, image_size, decoder='sbd'):
        super(SpatialBroadcastDec, self).__init__()
        self.height = image_size[0]
        self.width = image_size[1]
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim+2, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result
