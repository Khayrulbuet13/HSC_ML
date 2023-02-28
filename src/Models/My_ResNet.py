import torch
torch.cuda.empty_cache()
from torch import nn
import torchvision.models as models


class ResNet_mod:
    
    def __init__(self, model_type = "resnet50", weight = "IMAGENET1K_V2", num_class = 2):
        
        self.model_type = model_type
        self.weight     = weight
        self.num_class  = num_class
        self.model  = self.get_model()

        return self.model
        
    def get_model(self):
        if self.model_type == "resnet50":
            model = models.resnet50(weights = self.weight)
            
            # get the input feature in fully connected layer
            num_in_features = model.fc.in_features
            
            # Replace the final fully connected layer to suite the problem
            model.fc = nn.Sequential(nn.Linear(num_in_features, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(512, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 32),
                                            nn.Softmax(),
                                            nn.Linear(32, self.num_class))
        #return model


class CNN_AE(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256), dec_chs=(256, 128, 64, 3)):
        pass


    def forward(self, x):
        pass

# Test the network

# x = torch.randn(10, 3, 64, 64)
# enc_block_ck = CNN_AE()
# ftrs = enc_block_ck(x)
# for ftr in ftrs: print(ftr.shape)

# model = ResNet_mod("resnet50", weight = "IMAGENET1K_V2", num_class = 2)
# print(model)