import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device
        self.action_size = action_size

        # Define the neural network layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)  # Output: 100x100
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # Output: 50x50
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Output: 25x25
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Output: 13x13
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # Output: 7x7

        # Fully connected layers
        self.fc1 = nn.Linear(10 * 8 * 512, 1024)
        self.fc2 = nn.Linear(1024, action_size)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values
        """
        if isinstance(observation, np.ndarray):  # If observation is a NumPy array, convert to Tensor
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)


            # Normalize observation values to [0, 1]
        x = observation / 255.0

        # Reshape input to match NCHW format (Batch, Channels, Height, Width)
        if x.ndim == 3:  # If input is a single image (Height, Width, Channels)
            x = x.unsqueeze(0)  # Add batch dimension
        if x.shape[1] != 3:  # If channels dimension is not 3, assume NHWC format and permute
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Apply convolutional layers with ReLU activation
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        # Flatten the output of the conv layers
        x = x.contiguous().view(x.size(0), -1)

        # Apply fully connected layers
        x = F.leaky_relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values
