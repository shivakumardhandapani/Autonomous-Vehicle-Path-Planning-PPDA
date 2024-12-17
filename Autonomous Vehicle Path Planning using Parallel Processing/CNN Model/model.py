import torch.nn as nn


class steeringPredictionModel(nn.Module):
    """
    Prediction Model inspired from https://arxiv.org/pdf/1604.07316 

    Model w/ 9 layers
     - CNN: 5 convolutional layers with an activation function at each layer
     - FC: 4 fully connected layers with an output layer to predict steering angle
    """
    def __init__(self, inputImageSize):
        super(steeringPredictionModel,self).__init__()

        def _calc_flattened_feature_size(imageSize):
            h,w = imageSize
            layers = [(5,2), (5,2), (5,2), (3,1), (3,1)]

            for kern, stride in layers:
                h = (h - kern) // stride + 1
                w = (w - kern) // stride + 1
            
            return h*w # squarred for Height * Width of image
        

        self.featureSize = _calc_flattened_feature_size(inputImageSize)  

        # CNN Block for model
        self.cnnBlock = nn.Sequential(
            nn.Conv2d(3,24,kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24,36,kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36,48,kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48,64,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3),
            nn.ReLU()
        )

        # Fully Connected Block
        self.fcBlock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*self.featureSize, 1164), #64 output from CNN, imageSize = LxW of image
            nn.ReLU(),
            nn.Linear(1164,100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10),
            nn.ReLU(),
            nn.Linear(10,1) #output steerign angle prediction
        )


    def forward(self, inputImageTensor):
        cnnOut = self.cnnBlock(inputImageTensor)
        prediction = self.fcBlock(cnnOut)

        return prediction