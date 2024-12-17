import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import logging
from resnetmodel import ResNetSteeringModel
from loader import SteeringImageData
from utils import calculate_rmse

def main():
    # Configure logging
    logging.basicConfig(
        filename="training_parallel.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Check for GPUs and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load dataset and apply transformations
    dataset = SteeringImageData(dir="Dataset/Data", trans=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))

    # Split dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Load data with DataLoader and parallel-friendly settings
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=8 * torch.cuda.device_count(), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=8 * torch.cuda.device_count(), pin_memory=True
    )

    logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # Initialize model and wrap with DataParallel if multiple GPUs are available
    model = ResNetSteeringModel()
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)  # Move model to device

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for epoch in range(10):
        model.train()
        epoch_train_loss = 0.0
        logging.info(f"Epoch {epoch + 1} training started")

        for images, angles in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            images, angles = images.to(device), angles.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output.squeeze(), angles)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} completed. Avg Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for images, angles in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
                images, angles = images.to(device), angles.to(device)
                output = model(images)
                val_loss += criterion(output.squeeze(), angles).item()
                val_rmse += calculate_rmse(output.squeeze().cpu().numpy(), angles.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        logging.info(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}, Val RMSE = {avg_val_rmse:.4f}")
        print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}, Val RMSE = {avg_val_rmse:.4f}")

    end_event.record()
    torch.cuda.synchronize()  # Wait for events to complete
    training_time = start_event.elapsed_time(end_event) / 1000
    logging.info(f"Training time: {training_time:.2f} seconds")

    # Save the trained model
    model_path = "parallel_results_best_model.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
