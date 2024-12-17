import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score
import logging
from loader import SteeringImageData
from model import steeringPredictionModel

# Configure logging
logging.basicConfig(
    filename="testing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info("Starting testing process")

# Paths
TEST_DATA_DIR = r"E:\data\testing_data\ClearNoon"  # Update to your test dataset directory
MODEL_PATH = r"C:\Users\Derek\Documents\Northeastern\PP_DA\FinalProject\singleCore\steering_angle_model.pth" # Path to the saved model
# MODEL_PATH = r"C:\Users\Derek\Documents\Northeastern\PP_DA\FinalProject\finalMulti\steering_angle_model.pth" # Path to the saved model

# Parameters
BATCH_SIZE = 32

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((220, 220)),
    transforms.ToTensor()
])

# Load Dataset
test_dataset = SteeringImageData(dir=TEST_DATA_DIR, trans=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

image, _ = test_dataset[0]
imageSize = image.data.shape[1:]

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
model = steeringPredictionModel(inputImageSize=imageSize).to(device)

# Load the trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
logging.info(f"Loaded model weights from {MODEL_PATH}")

# Metrics storage
all_targets = []
all_predictions = []

# Testing loop
logging.info("Starting evaluation")
with torch.no_grad():
    for images, angles in test_loader:
        images, angles = images.to(device), angles.to(device)
        predictions = model(images).squeeze()
        all_targets.extend(angles.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Convert to tensors for metric calculations
all_targets = torch.tensor(all_targets)
all_predictions = torch.tensor(all_predictions)


# Evaluate metrics
mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

# Log and print results
log_message = (
    f"Test Results:\n"
    f"Mean Squared Error (MSE): {mse:.4f}\n"
    f"Mean Absolute Error (MAE): {mae:.4f}\n"
    f"R² Score: {r2:.4f}\n"
)
logging.info(log_message)
print(log_message)

# Save the results
results_path = "test_results.txt"
with open(results_path, "w") as f:
    f.write(log_message)

print(f"Test results saved to {results_path}")
logging.info("Testing process completed")
