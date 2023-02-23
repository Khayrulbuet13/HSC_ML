import torch
torch.cuda.empty_cache()
from torch import nn


class En_Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_features),
            nn.MaxPool2d(2, stride=2)
        )
    
    def forward(self, x):
        x = self.block(x)
        return x


class Dec_Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_features),
            nn.UpsamplingNearest2d(scale_factor=2)
        )
    
    def forward(self, x):
        x = self.block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([En_Block(chs[i], chs[i+1], kernel_size=3, padding=1) 
                                         for i in range(len(chs)-1)])
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
        return ftrs
    
class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Dec_Block(chs[i], chs[i+1], kernel_size=3, padding=1) 
                                         for i in range(len(chs)-1)])
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
        return ftrs   
    


class CNN_AE(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256), dec_chs=(256, 128, 64, 3)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)


    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0])

        return out


# Test the network

# x = torch.randn(10, 3, 64, 64)
# enc_block_ck = CNN_AE()
# ftrs = enc_block_ck(x)
# for ftr in ftrs: print(ftr.shape)
