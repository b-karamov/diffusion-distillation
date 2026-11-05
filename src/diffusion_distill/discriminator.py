"""CNN-дискриминатор и состязательные hinge-потери."""

from torch import nn
from torch.nn import functional as F


class SmallDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        layers = []
        channels = 3
        for width in (base_channels, base_channels * 2, base_channels * 4):
            layers.extend(
                [nn.Conv2d(channels, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
            )
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, image):
        return self.head(self.features(image).mean(dim=(2, 3))).flatten()


def discriminator_loss(discriminator, real, fake):
    return (
        F.relu(1 - discriminator(real)).mean()
        + F.relu(1 + discriminator(fake.detach())).mean()
    )


def generator_loss(discriminator, fake):
    return -discriminator(fake).mean()
