import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import EyeDataset, ValDataset, load_dataset_paths, load_ground_truths
from src.algorithms.neural_network import get_efficientnet_model
from src.visualization import plot_history

def train_and_validate(model, train_loader, val_loader, n_epochs, device):
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_mse(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                loss = loss_mse(pred, labels)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        history['train'].append(avg_train)
        history['val'].append(avg_val)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

    return history

def main():
    parser = argparse.ArgumentParser(description="Training EfficientNet for circle detection.")
    parser.add_argument("--img_dir", type=str, default="data/circle", help="Path to the image directory.")
    parser.add_argument("--csv_file", type=str, default="ground_truths.csv", help="Path to the CSV file containing pseudo-ground truths.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    
    args = parser.parse_args()
    img_dir = Path(args.img_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loading data
    print("Loading file paths...")
    _, val_set = load_dataset_paths(img_dir)
    ground_truths = load_ground_truths(img_dir)
    
    print("Preparing datasets...")
    train_dataset = EyeDataset(csv_file=args.csv_file)
    val_dataset = ValDataset(val_paths=val_set, ground_truths=ground_truths)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model Initialization
    print("Creating the EfficientNet model...")
    model = get_efficientnet_model(pretrained=True)
    model.to(device)
    
    # Training Loop
    print("Starting training...")
    history = train_and_validate(model, train_loader, val_loader, n_epochs=args.epochs, device=device)
    
    # Plotting Learning Curves
    plot_history(history)
    
    # Saving Weights
    model_path = "efficientnet_eyes.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")

if __name__ == "__main__":
    main()
