import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # Convert state to tensor and move to the appropriate device
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Ensure state_tensor has the correct number of dimensions
    if state_tensor.dim() == 2:
        state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif state_tensor.dim() == 3:
        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

    state_tensor = state_tensor.permute(0, 3, 1, 2).to(next(policy_net.parameters()).device) / 255.0

    # Compute Q-values using the policy network
    with torch.no_grad():
        q_values = policy_net(state_tensor)

    # Ensure the number of Q-values matches the action size
    assert q_values.shape[1] == action_size, f"Expected {action_size} actions, but got {q_values.shape[1]} Q-values."

    # Select the action with the highest Q-value
    action = torch.argmax(q_values, dim=1).item()
    # print("This is the greedy action", action)
    return action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    epsilon = exploration.value(t)

    if random.random() < epsilon:  # Explore
        action = random.randint(0, action_size - 1)
    else:  # Exploit
        action = select_greedy_action(state, policy_net, action_size)


    # print("THis is an exploratory action", action)
    return action

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [-0.5, 0.05, 0], [0.5, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0], [0, 0, 0]]
