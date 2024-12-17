import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import logging
from resnetmodel import ResNetSteeringModel
from loader import SteeringImageData
from utils import calculate_rmse
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info("Starting the training process")

# Load and split dataset
dataset = SteeringImageData(dir="Dataset/Data", trans=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
]))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Model
model = ResNetSteeringModel().to(device)  # Move model to GPU
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    epoch_train_loss = 0.0

    logging.info(f"Epoch {epoch + 1} training started")
    for images, angles in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        images, angles = images.to(device), angles.to(device)  # Ensure data is moved to GPU
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output.squeeze(), angles)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    logging.info(f"Epoch {epoch + 1} training completed. Average Loss = {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    val_rmse = 0.0

    logging.info(f"Epoch {epoch + 1} validation started")
    with torch.no_grad():
        for images, angles in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
            images, angles = images.to(device), angles.to(device)  # Ensure data is moved to GPU
            output = model(images)
            val_loss += criterion(output.squeeze(), angles).item()
            val_rmse += calculate_rmse(output.squeeze().cpu().numpy(), angles.cpu().numpy())  # Compute RMSE for validation

    avg_val_loss = val_loss / len(val_loader)
    avg_val_rmse = val_rmse / len(val_loader)

    logging.info(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}, Validation RMSE = {avg_val_rmse:.4f}")
    print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}, Validation RMSE = {avg_val_rmse:.4f}")

# Save the model
model_path = "results_best_model.pth"
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")
logging.info("Training process completed")
