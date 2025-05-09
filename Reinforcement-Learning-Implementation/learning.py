import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """

    # Step 1: Sample transitions from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to tensors and move to device
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)  # Add dimension for gathering
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device).unsqueeze(1)  # Add dimension

    # Step 2: Compute Q(s_t, a)
    q_values = policy_net(states).gather(1, actions)

    # Step 3: Compute max_a Q(s_{t+1}, a) using the target network
    with torch.no_grad():  # No gradient for target network
        max_next_q_values = target_net(next_states).max(1, keepdim=True)[0]

    # Step 4: Mask next state values where episodes have terminated
    max_next_q_values[dones] = 0.0

    # Step 5: Compute the target
    target_q_values = rewards + gamma * max_next_q_values

    # Step 6: Compute the loss
    loss = F.mse_loss(q_values, target_q_values)

    # Step 7: Calculate the gradients
    optimizer.zero_grad()
    loss.backward()

    # Step 8: Clip the gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    # Step 9: Optimize the model
    optimizer.step()

    return loss.item()


def perform_double_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a double Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # Step 1: Sample transitions from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to tensors and move to device
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)  # Add dimension for gathering
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device).unsqueeze(1)  # Add dimension

    # Step 2: Compute Q(s_t, a)
    q_values = policy_net(states).gather(1, actions)

    # Step 3: Double Q-learning: Use policy network to select actions, target network to evaluate
    with torch.no_grad():  # No gradient for target network
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        max_next_q_values = target_net(next_states).gather(1, next_actions)

    # Step 4: Mask next state values where episodes have terminated
    max_next_q_values[dones] = 0.0

    # Step 5: Compute the target
    target_q_values = rewards + gamma * max_next_q_values

    # Step 6: Compute the loss
    loss = F.mse_loss(q_values, target_q_values)

    # Step 7: Calculate the gradients
    optimizer.zero_grad()
    loss.backward()

    # Step 8: Clip the gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    # Step 9: Optimize the model
    optimizer.step()

    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network

    target_net.load_state_dict(policy_net.state_dict())