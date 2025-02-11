import time
import random
import argparse
import numpy as np
import torch

from network import ClassificationNetwork
from dataset import get_dataloader

def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    gpu = torch.device('cuda')

    nr_epochs = 100
    batch_size = 64
    nr_of_classes = 9  # needs to be changed
    start_time = time.time()
    
    train_loader = get_dataloader(data_folder, batch_size)

    infer_action.to(gpu)

    # List to store the total loss for each epoch
    epoch_losses = []

    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)

            batch_out = infer_action(batch_in)
            loss = cross_entropy_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        # Average loss for the current epoch
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)  # Record the average loss for this epoch

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))


    # After training, plot the losses
    plot_loss(epoch_losses)


    torch.save(infer_action, save_path)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    # log_probs = torch.log(batch_out + 1e-10) 
    # loss = -torch.mean(torch.sum(batch_gt * log_probs, dim=1))
    # return loss

    batch_gt = batch_gt.long()

    # Code found from geeksforgeeks.com
    # Computing element wise exponential value
    exp_values = torch.exp(batch_out)

    # Computing sum of these values
    exp_values_sum = torch.sum(exp_values, dim=1, keepdim=True)

    # Calculating the softmax for training data.
    softmax = exp_values/exp_values_sum
    
    # Compute the log of softmax
    log_probs = torch.log(softmax + 1e-10)  # Adding small value to prevent log(0)

    # Gather the log probabilities corresponding to the ground truth classes
    batch_size = batch_out.size(0)
    loss = -log_probs[range(batch_size), batch_gt[0]]

    # Step 4: Take the mean of the losses over the batch
    loss = torch.mean(loss)
 
    return loss


def plot_loss(epoch_losses):
    """
    Plots the loss curve over the epochs.
    """
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC518 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Training_Data", type=str, help='C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Training_Data/')
    parser.add_argument('-s', '--save_path', default="C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Trained Model", type=str, help='C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Trained Model')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)