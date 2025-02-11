import torch
import numpy as np
from torch.utils.data import DataLoader
from MyDataloader import MyDataLoader
from HeavyModelResNet101 import HeavyModelResNet101
import matplotlib.pyplot as plt

def load_model(model_path):
    checkpoint = torch.load(model_path)  # Load the entire checkpoint
    model = HeavyModelResNet101()  # Initialize your model
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

    model.load_state_dict(state_dict)  # Load only the model state dict
    model.eval()  # Set the model to evaluation mode
    return model

def plot_predictions_vs_ground_truth(predictions, ground_truths):
    # creating initial predictions and gt
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted Throttle", marker='o', linestyle='-', color='blue')
    plt.plot(ground_truths, label="Ground Truth Throttle", marker='x', linestyle='--', color='red')

    # Add labels, legend, and title
    plt.xlabel("Sample Index")
    plt.ylabel("Throttle Value")
    plt.title("Predicted Throttle vs Ground Truth")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_predictions_vs_ground_truth_with_averages(predictions, ground_truths):
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Compute averages
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truths)

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label="Predicted Throttle", marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.plot(ground_truths, label="Ground Truth Throttle", marker='x', linestyle='--', color='red', alpha=0.7)

    # Add horizontal lines for averages
    plt.axhline(mean_pred, color='blue', linestyle=':', label=f"Mean Prediction: {mean_pred:.2f}")
    plt.axhline(mean_gt, color='red', linestyle=':', label=f"Mean Ground Truth: {mean_gt:.2f}")

    # Add labels, legend, and title
    plt.xlabel("Sample Index")
    plt.ylabel("Throttle Value")
    plt.title("Predicted Throttle vs Ground Truth (with Averages)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def print_predictions(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, throttle in dataloader:
            images = images.to(device)
            predictions = model(images)
            predictions = (predictions >= 0.5).float()  # CLIPPING THE PREDICTIONS
            predictions = predictions.cpu().numpy()

            all_predictions.extend(predictions)
            all_ground_truths.extend(throttle.numpy())

    # Print predictions and ground truth
    for i, (prediction, ground_truth) in enumerate(zip(all_predictions, all_ground_truths)):
        print(f"Sample {i}: Predicted Throttle: {prediction[0]}, Ground Truth Throttle: {ground_truth[0]}")

    # just ground truth vs prediction
    plot_predictions_vs_ground_truth(all_predictions, all_ground_truths)

    # ground truth vs prediction with averages
    plot_predictions_vs_ground_truth_with_averages(all_predictions, all_ground_truths)


def get_preds(model_path, run_dir, batch_size=16):
    # Load the model
    model = load_model(model_path)

    # Create DataLoader for the new dataset
    test_dataset = MyDataLoader(run_dir)  # our dataset that we use for testing
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print predictions
    return print_predictions(model, dataloader)

if __name__ == "__main__":
    model_path = "C:/Users/Kote/OneDrive/BU Homework/Robot Learning/Final Project/Imitation Learning/heavymodel.pth"  # Model changer
    run_dir = "C:/Users/Kote/OneDrive/BU Homework/Robot Learning/Final Project/Imitation Learning/"
    get_preds(model_path, run_dir)
