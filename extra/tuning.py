import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from model import steeringPredictionModel
from loader import SteeringImageData
import numpy as np

torch.set_num_threads(1)

# Transforms for training and validation
trainTransform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

valTransform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = SteeringImageData(dir="4/Data", trans=trainTransform)
image, _ = dataset[0]
imageSize = image.data.shape

# Split into training and validation datasets
trainSize = int(0.8 * len(dataset))
valSize = len(dataset) - trainSize
trainDataset, valDataset = random_split(dataset, [trainSize, valSize])
valDataset.dataset.transform = valTransform

# Hyperparameter options
learning_rates = [0.0001, 0.001, 0.01, 0.1]
batch_sizes = [512, 1024, 2048]
optimizers = ['adam', 'sgd']

best_rmse = float('inf')
best_params = {}

# Training and evaluation loop for hyperparameter tuning
for lr in learning_rates:
    for batch_size in batch_sizes:
        for opt in optimizers:
            print(f"\nTraining with lr={lr}, batch_size={batch_size}, optimizer={opt}")

            trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
            valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False)

            device = torch.device("cpu")
            model = steeringPredictionModel(inputImageSize=imageSize[1]).to(device)
            criterion = nn.MSELoss()
            optimizer = (optim.Adam if opt == 'adam' else optim.SGD)(model.parameters(), lr=lr)

            num_epochs = 1
            for epoch in range(num_epochs):
                model.train()
                trainLoss = 0.0
                with tqdm(enumerate(trainLoader), desc=f"Epoch {epoch}", unit="iter", total=len(trainLoader)) as pbar:
                    for iteration, (images, angles) in pbar:
                        images, angles = images.to(device), angles.to(device)
                        predictions = model(images)
                        loss = criterion(predictions.squeeze(), angles)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        trainLoss += loss.item()
                        pbar.set_postfix(loss=loss.item())

                # Validation phase
                model.eval()
                valLoss = 0.0
                with torch.no_grad():
                    for images, angles in valLoader:
                        images, angles = images.to(device), angles.to(device)
                        predictions = model(images)
                        loss = criterion(predictions.squeeze(), angles)
                        valLoss += loss.item()

                rmse = np.sqrt(valLoss / len(valLoader))
                print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {trainLoss / len(trainLoader):.4f}, RMSE: {rmse:.4f}")

            # Track the best parameters
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'learning_rate': lr, 'batch_size': batch_size, 'optimizer': opt}
                torch.save(model.state_dict(), "best_steering_angle_model.pth")

print(f"\nBest Hyperparameters: {best_params} with RMSE: {best_rmse:.4f}")
print("Best model saved as best_steering_angle_model.pth")
