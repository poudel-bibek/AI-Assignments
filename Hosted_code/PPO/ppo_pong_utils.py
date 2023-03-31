import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# Constants for actions
RIGHT = 4
LEFT = 5

# Set the device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_single(image, bkg_color=np.array([144, 72, 17])):
    """
    Preprocess a single image frame by cropping, downscaling, and normalizing.
    """
    img = np.mean(image[34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    return img

def preprocess_batch(images, bkg_color=np.array([144, 72, 17])):
    """
    Preprocess a batch of images for input to a PyTorch neural network.
    """
    list_of_images = np.asarray(images)
    
    # Ensure the correct number of dimensions
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    
    # Subtract background color, crop, and normalize
    list_of_images_prepro = np.mean(list_of_images[:, :, 34:-16:2, ::2] - bkg_color, axis=-1) / 255.
    
    # Swap axes for proper input format
    batch_input = np.swapaxes(list_of_images_prepro, 0, 1)
    
    return torch.from_numpy(batch_input).float().to(device)

def collect_trajectories(envs, policy, tmax=200, nrand=5):
    """
    Collect trajectories for an environment using a given policy.
    """
    n = len(envs.ps)  # Number of parallel instances

    # Initialize lists for storing results
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    envs.reset()

    # Start all parallel agents
    envs.step([1] * n)

    # Perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _, _ = envs.step(np.random.choice([RIGHT, LEFT], n))
        fr2, re2, _, _, _ = envs.step([0] * n)  # Advance game 1 frame by doing nothing.

    for t in range(tmax):
        # Prepare input for policy
        batch_input = preprocess_batch([fr1, fr2])

        # Calculate action probabilities (no gradient needed, move to CPU)
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        # Determine actions based on probabilities
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action == RIGHT, probs, 1.0 - probs)

        # Advance the game by taking one action and skipping one frame
        fr1, re1, is_done, _, _ = envs.step(action)
        fr2, re2, is_done, _, _ = envs.step([0] * n)

        # Calculate reward
        reward = re1 + re2

        # Store results
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        # Stop if any trajectory is done (to keep lists rectangular)
        if is_done.any():
            break

    # Return action probabilities, states, actions, and rewards
    return prob_list, state_list, action_list, reward_list

def states_to_prob(policy, states):
    """
    Convert states to probabilities using a given policy.
    """
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])

    # Compute the probabilities of the states using the policy
    return policy(policy_input).view(states.shape[:-3])