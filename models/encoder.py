import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, image_size):
        super(Encoder, self).__init__()
        height = image_size[0]
        width = image_size[1]
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        return self.mlp(x)


class GaussianParamNet(nn.Module):
    """
    Parameterise a Gaussian distributions.
    """
    def __init__(self, input_dim, output_dim):
        super(GaussianParamNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.layer_nml = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: input image with shape [B, K, 2*D]
        """
        # obtain size of []
        x = self.fc2(F.relu(self.layer_nml(self.fc1(x))))
        mu, sigma = x.chunk(2, dim=-1)
        sigma = F.softplus(sigma + 0.5) + 1e-8
        return mu, sigma
