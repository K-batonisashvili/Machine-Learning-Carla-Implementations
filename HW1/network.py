import torch
import torch.nn as nn

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 320x240 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')

        super(ClassificationNetwork, self).__init__()

        # Convolution part
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        #Fully connected part
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 38 * 28, 512),  # Adjusted input size
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 9),  # 9 classes for the actions
            nn.Softmax(dim=1)
        )


    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        y = self.conv_layers(observation)

        y = y.view(y.size(0), -1)

        y = self.fc_layers(y)

        return y

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """

        classes = []
        for action in actions:
            throttle, steering, brake = action[0], action[1], action[2]

            if throttle > 0:  # If throttle is applied
                if steering < -0.01:  # Steer left
                    class_idx = 3  # Steer Left and Throttle
                elif steering > 0.01:  # Steer right
                    class_idx = 4  # Steer Right and Throttle
                else:
                    class_idx = 5  # Steer Straight and Throttle
            elif brake > 0:  # If brake is applied
                if steering < -0.01:  # Steer left
                    class_idx = 6  # Steer Left and Brake
                elif steering > 0.01:  # Steer right
                    class_idx = 7  # Steer Right and Brake
                else:
                    class_idx = 8  # Steer Straight and Brake
            else:  # No throttle or brake so steering only
                if steering < -0.01:  # Steer left
                    class_idx = 0  # Steer Left
                elif steering > 0.01:  # Steer right
                    class_idx = 1  # Steer Right
                else:
                    class_idx = 2  # Steer Straight

            # One-hot encode
            one_hot = torch.zeros(9)  # 9 classes
            one_hot[class_idx] = 1
            classes.append(one_hot)

        return classes

    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """

        class_idx = torch.argmax(scores).item()

        '''
        throttle, steering, brake
        throttle is between 0 and 1
        brake is between 0 and 1
        steering is between -1 and 1        
        '''

        if class_idx == 0:  # Steer Left
            return (0.0, -1.0, 0.0)
        elif class_idx == 1:  # Steer Right
            return (0.0, 1.0, 0.0)
        elif class_idx == 2:  # Steer Straight
            return (0.0, 0.0, 0.0)
        elif class_idx == 3:  # Steer Left and Throttle
            return (1.0, -1.0, 0.0)
        elif class_idx == 4:  # Steer Right and Throttle
            return (1.0, 1.0, 0.0)
        elif class_idx == 5:  # Steer Straight and Throttle
            return (1.0, 0.0, 0.0)
        elif class_idx == 6:  # Steer Left and Brake
            return (0.0, -1.0, 1.0)
        elif class_idx == 7:  # Steer Right and Brake
            return (0.0, 1.0, 1.0)
        elif class_idx == 8:  # Steer Straight and Brake
            return (0.0, 0.0, 1.0)


