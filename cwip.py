import torch
from torch import nn
import torchvision

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape, torch.prod(torch.tensor(x.shape)))
        return x

class Transpose(nn.Module):
    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)

class ConvBlock2d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2d, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

class ConvBlock1d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1d, self).__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

class CWIPModel(torch.nn.Module):
    def __init__(self):
        super(CWIPModel, self).__init__()

        self.img_block = nn.Sequential(
            torchvision.transforms.Resize((320,640)),
            ConvBlock2d(4, 16), 
            ConvBlock2d(16, 64),
            ConvBlock2d(64, 128),
            ConvBlock2d(128, 256),
            nn.Flatten(-3),
            nn.Linear(256*20*40, 256*312),
            nn.Sigmoid(),
        )

        self.wav_block = nn.Sequential(
            ConvBlock1d(12, 16),
            ConvBlock1d(16, 64),
            ConvBlock1d(64, 128),
            ConvBlock1d(128, 256),
            nn.Flatten(-2),
        )

    def forward(self, img, wav):
        img_rep = self.img_block(img)
        wav_rep = self.wav_block(wav)
        out = torch.tensordot(img_rep, wav_rep, dims=([1], [1]))
        return out / torch.max(out)

