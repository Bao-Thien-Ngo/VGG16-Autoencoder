import torch
import torchvision
import torch.nn as nn
import numpy as np
import os

# Initialize batch_size and number of epoch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Autoencoder model
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),  # 32,224,224
            nn.ReLU(),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),  # 64,112,112
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),  # 64,112,112
            nn.ReLU(),
            nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3), padding=1),  # 128,56,56
            nn.ReLU(),
            nn.Conv2d(128, 128, stride=(1, 1), kernel_size=(3, 3), padding=1),  # 128,56,56
            nn.ReLU(),
            nn.Conv2d(128, 256, stride=(2, 2), kernel_size=(3, 3), padding=1),  # 256,28,28
            nn.ReLU()
        )

        self.feature_extractor = nn.Conv2d(256, 256, stride=(1, 1), kernel_size=(3, 3), padding=1)  # 256,28,28

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, stride=(2, 2), kernel_size=(2, 2)),  # 128,56,56
            nn.Conv2d(128, 128, stride=(1, 1), kernel_size=(3, 3), padding=1),  # 128,56,56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, stride=(2, 2), kernel_size=(2, 2)),  # 64,112,112
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),  # 64,112,112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(2, 2)),  # 32,224,224
            nn.Conv2d(32, 3, stride=(1, 1), kernel_size=(3, 3), padding=1),  # 3,224,224
        )

    def forward(self, x):
        x = self.encoder(x)
        extract_feature = self.feature_extractor(x)
        y = self.decoder(extract_feature)
        return y, extract_feature


# Build the model
model = AutoEncoder()
model.to(DEVICE)



