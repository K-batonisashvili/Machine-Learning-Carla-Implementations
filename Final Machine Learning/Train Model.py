import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
import argparse
from HeavyModelResNet101 import HeavyModelResNet101  # we change this for small model vs big model
import logging
from MyDataloader import MyDataLoader


def train(data_folder, save_path):
    # parameters are adjusted based on which model is loaded
    device = torch.device('cuda')
    nr_epochs = 2024
    batch_size = 64
    start_time = time.time()
    l1_lambda = 0.001
    early_stop_patience = 300 # Number of epochs to wait for improvement before stopping

    # data loader for training which parses through our directories and finds respective rgb, depth, action
    try:
        full_dataset = MyDataLoader(data_folder)
        full_size = len(full_dataset)

        # split dataset into training (80%) and validation (20%)
        train_size = int(0.8 * full_size)
        val_size = full_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # data loader creation for validation and training separately
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # initialize the model
        model = HeavyModelResNet101()
        model = nn.DataParallel(model)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        meansquareloss = nn.MSELoss()

        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        loss_values = []
        val_loss_values = []  # To track validation loss
        best_val_loss = float('inf')
        epochs_without_improvement = 0 # initialize variable to keep track of improvements

        for epoch in range(nr_epochs):
            total_loss = 0

            # Training phase
            model.train()
            for batch_idx, (batch_in, batch_gt) in enumerate(train_loader):
                batch_in = batch_in.to(device)
                batch_gt = batch_gt.to(device)

                # Forward pass
                optimizer.zero_grad()
                batch_out = model(batch_in)
                loss = meansquareloss(batch_out, batch_gt)

                # L1 Regularization
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss += l1_lambda * l1_norm

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / (batch_idx + 1)
            loss_values.append(average_loss)
            scheduler.step()

            # Validation phase
            model.eval()
            val_total_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch_in, val_batch_gt = val_batch
                    val_batch_in = val_batch_in.to(device)
                    val_batch_gt = val_batch_gt.to(device)

                    val_outputs = model(val_batch_in)
                    val_loss = meansquareloss(val_outputs, val_batch_gt)
                    val_total_loss += val_loss.item()

            average_val_loss = val_total_loss / len(val_loader)
            val_loss_values.append(average_val_loss)

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                epochs_without_improvement = 0
                # Save the best model checkpoint
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': average_val_loss,
                }
                torch.save(best_checkpoint, save_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stop_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

            time_per_epoch = (time.time() - start_time) / (epoch + 1)
            time_left = time_per_epoch * (nr_epochs - 1 - epoch)
            logging.info(
                f"Epoch {epoch + 1}\t[Train]\tloss: {average_loss:.6f} \t[Val] loss: {average_val_loss:.6f} \tETA: +{time_left:.2f}s")

        # Plot loss values
        plt.figure()
        plt.title('Loss Plot for the Heavy Model')
        plt.plot(loss_values, label='Training Loss')
        plt.plot(val_loss_values, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Loss_Plot.png')
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    # THE FOLDER PATHS NEED TO BE CHANGED IN ACCORDANCE TO WHERE THEY ARE SAVED.
    # Current path is using my personal folders
    parser = argparse.ArgumentParser(description='Imitation Learning Training Script')
    parser.add_argument('-d', '--data_folder',
                        default="C:/Users/Kote/OneDrive/BU Homework/Robot Learning/Final Project/Imitation Learning/",
                        type=str, help='Path to your dataset')
    parser.add_argument('-s', '--save_path',
                        default="C:/Users/Kote/OneDrive/BU Homework/Robot Learning/Final Project/Imitation Learning/heavymodel.pth",
                        type=str, help='Path to save your model')
    args = parser.parse_args()

    train(args.data_folder, args.save_path)