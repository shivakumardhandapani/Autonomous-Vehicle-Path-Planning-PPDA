import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SteeringImageData(Dataset):
    def __init__(self, dir, trans=None):
        # self.imageDir = os.path.join(dir, "") For Testing only
        # self.steerFile = os.path.join(dir, "steer_values.txt") #use this fo rtesting
        self.steerFile = os.path.join(dir, "SteerValues\steer_values.txt")  #use this for training
        self.imageDir = os.path.join(dir, "Images")
        
        self.transform = trans

        with open(self.steerFile, 'r') as fid:
            self.steerVals = [float(line.strip()) for line in fid]

        self.imageFileNames = [f"image{str(ii).zfill(9)}.png" for ii in range(len(self.steerVals))]
        # self.imageFileNames = [f"ClearNoon{str(ii).zfill(9)}.png" for ii in range(len(self.steerVals))] #TESTING ONLY

        for fileName in self.imageFileNames:
            if not os.path.isfile(os.path.join(self.imageDir, fileName)):
                raise FileNotFoundError(f"Image File not found: {fileName} in {self.imageDir}")
            
    def __len__(self):
        return len(self.imageFileNames)
    
    def __getitem__(self, index):
        imgFullFile = os.path.join(self.imageDir, self.imageFileNames[index])
        img = Image.open(imgFullFile).convert("RGB")
        if self.transform:
            img = self.transform(img)

        steeringAngle = torch.tensor(self.steerVals[index], dtype=torch.float32)

        return img, steeringAngle