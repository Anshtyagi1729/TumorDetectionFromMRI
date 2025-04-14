import torch
import torch.nn as nn
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),  #input: 1,128,128
            nn.ReLU(),
            nn.MaxPool2d(2),  #32,64,64

            nn.Conv2d(32,64,3,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), #64,32,32

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) #128,16,16
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,4) #4 classes here
        )
    def forward(self,x):
        x=self.net(x)
        return self.classifier(x)
def load_model():
    model=BrainTumorCNN()
    model.load_state_dict(torch.load('model.pth',map_location="cpu"))
    model.eval()
    return model
