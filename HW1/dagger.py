import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from network import ClassificationNetwork
from dataset import get_dataloader


def expert_policy(state):
    """
    Simulates an expert policy that outputs the correct action for the given state.
    In real implementations, this would be a pre-trained model or a hard-coded policy.
    """
    # Dummy expert action - replace this with the actual expert policy or network
    return torch.randint(0, 9, (state.size(0),))  # assuming 9 classes


def train_DAgger(data_folder, save_path, N=10):
    """
    Implements the DAgger algorithm and applies it to the dataset.
    """
    # Initialize the classifier and the optimizer
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    gpu = torch.device('cuda')

    batch_size = 64
    nr_of_classes = 9
    train_loader = get_dataloader(data_folder, batch_size)

    infer_action.to(gpu)

    D = []  # Initialize dataset D
    β = lambda i: max(0.1, 1 - (i / N))  # Decay schedule for mixing expert and learned policy

    regrets = []  # Store regret for plotting
    validation_losses = []  # Store validation losses for comparison

    # Run the DAgger loop
    for i in range(N):
        print(f"Iteration {i + 1}/{N}")
        Di = []  # Temporary dataset for this iteration
        total_loss = 0

        infer_action.train()  # Put the model in training mode

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)

            # Step 1: Sample T-step trajectories using a mixed policy
            with torch.no_grad():
                learned_actions = infer_action(batch_in)
                learned_policy_actions = torch.argmax(learned_actions, dim=1)

            expert_actions = expert_policy(batch_in).to(gpu)
            pi_i_actions = (β(i) * expert_actions + (1 - β(i)) * learned_policy_actions).long()

            # Step 2: Get the expert actions for states visited by mixed policy
            Di.extend([(batch_in[b], expert_actions[b]) for b in range(batch_in.size(0))])

            # Step 3: Compute loss and update learned policy
            optimizer.zero_grad()
            loss = F.cross_entropy(learned_actions, expert_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Step 4: Aggregate dataset D with Di
        D.extend(Di)

        # Step 5: Train classifier on aggregated dataset D
        D_states, D_labels = zip(*D)
        D_states = torch.stack(D_states)
        D_labels = torch.tensor(D_labels).long()

        # Re-train infer_action on the aggregated dataset
        for epoch in range(3):  # Limit epochs per iteration to avoid overfitting
            optimizer.zero_grad()
            predictions = infer_action(D_states.to(gpu))
            loss = F.cross_entropy(predictions, D_labels.to(gpu))
            loss.backward()
            optimizer.step()

        # Record regret as the difference between learned policy and expert
        regret = compute_regret(infer_action, expert_policy, train_loader, gpu)
        regrets.append(regret)

        print(f"Iteration {i + 1} completed. Loss: {total_loss:.4f}, Regret: {regret:.4f}")

    # Plot the regret
    plot_regret(regrets)

    # Save the final model
    torch.save(infer_action, save_path)


def compute_regret(infer_action, expert_policy, dataloader, device):
    """
    Compute the regret by measuring how much the learned policy's decisions differ
    from the expert's actions.
    """
    infer_action.eval()  # Put the model in evaluation mode
    regret = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_in, batch_gt = batch[0].to(device), batch[1].to(device)
            learned_actions = torch.argmax(infer_action(batch_in), dim=1)
            expert_actions = expert_policy(batch_in).to(device)

            # Regret is the number of mismatches between learned and expert actions
            mismatches = (learned_actions != expert_actions).sum().item()
            total_samples += batch_in.size(0)
            regret += mismatches

    # Normalize regret to [0, 1]
    return regret / total_samples


def plot_regret(regrets):
    plt.plot(range(1, len(regrets) + 1), regrets)
    plt.xlabel('Iteration')
    plt.ylabel('Regret')
    plt.title('Regret over Iterations of DAgger')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAgger Training for Classification')
    parser.add_argument('-d', '--data_folder',
                        default="C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Training_Data",
                        type=str)
    parser.add_argument('-s', '--save_path',
                        default="C:/Users/bcerv/OneDrive/Documents/Robot Learning/Robot_Learning_Hw1/Trained Model",
                        type=str)
    args = parser.parse_args()

    train_DAgger(args.data_folder, args.save_path)
