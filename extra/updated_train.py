import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from model import steeringPredictionModel
from loader import SteeringImageData
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import time


logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Transforms for training and validation
trainTransform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(15),     # Data augmentation
    transforms.ToTensor(),
])

valTransform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = SteeringImageData(dir="E:\data\Data", trans=trainTransform)
image, _ = dataset[0]
imageSize = image.data.shape

# Split into training and validation datasets
trainSize = int(0.8 * len(dataset))
valSize = len(dataset) - trainSize
trainDataset, valDataset = random_split(dataset, [trainSize, valSize])

# Apply validation-specific transform
valDataset.dataset.transform = valTransform

# Create DataLoaders
batch_size = 2048
trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss, and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = steeringPredictionModel(inputImageSize=imageSize[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 25
for epoch in range(num_epochs

    epoch_start_time = time.time()
                   
    model.train()  
    trainLoss = 0.0
    with tqdm(enumerate(trainLoader), desc=f"Training Progress Epoch {epoch}", unit="iter", total=len(trainLoader)) as pbar:
        for iteration, (images, angles) in pbar:
            images, angles = images.to(device), angles.to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions.squeeze(), angles)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()

            pbar.set_postfix(iter=iteration + 1, loss=loss.item())

    # Validation loop
    model.eval()  
    valLoss = 0.0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for images, angles in valLoader:
            images, angles = images.to(device), angles.to(device)
            predictions = model(images).squeeze()

            all_targets.extend(angles.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
        
            loss = criterion(predictions, angles)
            valLoss += loss.item()

    all_targets = torch.tensor(all_targets)
    all_predictions = torch.tensor(all_predictions)

    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    mse = valLoss / len(valLoader)

    epoch_end_time = time.time()  
    epoch_duration = epoch_end_time - epoch_start_time

    log_message = (f"Epoch [{epoch + 1}/{num_epochs}] - "
                   f"Training Loss: {trainLoss / len(trainLoader):.4f}, "
                   f"Validation Loss: {mse:.4f}, "
                   f"MAE: {mae:.4f}, R²: {r2:.4f}, "
                   f"Epoch Duration: {epoch_duration:.2f} seconds")

    logging.info(log_message)
    print(log_message)
    
# Save the trained model
torch.save(model.state_dict(), "steering_angle_model.pth")
logging.info("Model saved as steering_angle_model.pth")
print("Model saved as steering_angle_model.pth")
