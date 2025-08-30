# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 3 with LRTT: MNIST training using LRTT layers.

MNIST training example using LRTT (Low-Rank Tensor-Train) analog layers.
Based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""
# pylint: disable=invalid-name, redefined-outer-name

import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.rpu_base import cuda


# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 1
BATCH_SIZE = 64

# LRTT parameters - using different ranks for different layer sizes
LRTT_RANKS = [16, 16]  # Ranks for LRTT layers (input->hidden1, hidden1->hidden2)
TRANSFER_EVERY = 100  # Transfer A⊗B to C every N updates
LORA_ALPHA = 100.0  # LoRA scaling factor


def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def create_lrtt_config(rank):
    """Create LRTT configuration with specified rank.
    
    Args:
        rank (int): Rank for the LRTT decomposition
        
    Returns:
        PythonLRTTRPUConfig: LRTT configuration
    """
    device_config = PythonLRTTPreset.idealized(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA
    )
    device_config.correct_gradient_magnitudes = True  # Important for convergence
        # Configure transfer learning rate
    device_config.transfer_lr = device_config.lora_alpha
    
    return PythonLRTTRPUConfig(device=device_config)


def create_analog_network_lrtt(input_size, hidden_sizes, output_size):
    """Create the neural network using LRTT analog layers.

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model with LRTT
    """
    print(f"Creating LRTT network with ranks: {LRTT_RANKS} (final layer uses FloatingPoint)")
    
    model = AnalogSequential(
        # Layer 1: 784 -> 256 with rank 32
        AnalogLinear(
            input_size,
            hidden_sizes[0],
            bias=False,  # LRTT doesn't support bias
            rpu_config=create_lrtt_config(LRTT_RANKS[0]),
        ),
        nn.Sigmoid(),
        # Layer 2: 256 -> 128 with rank 16
        AnalogLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            bias=False,
            rpu_config=create_lrtt_config(LRTT_RANKS[1]),
        ),
        nn.Sigmoid(),
        # Layer 3: 128 -> 10 with FloatingPointDevice (final classification layer)
        AnalogLinear(
            hidden_sizes[1],
            output_size,
            bias=False,  # Can use bias with FloatingPoint
            rpu_config=FloatingPointRPUConfig(),
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    print(model)
    
    # Print LRTT configuration details
    print("\nLRTT Configuration:")
    print(f"  LRTT Ranks: {LRTT_RANKS} (layers 1-2)")
    print(f"  Final layer: FloatingPointDevice")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  Forward injection: enabled")
    print(f"  Gradient magnitude correction: enabled\n")
    
    return model


def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=0.05)
    optimizer.regroup_param_groups(model)

    return optimizer


def train(model, train_set):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    classifier = nn.NLLLoss()
    optimizer = create_sgd_optimizer(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    time_init = time()
    for epoch_number in range(EPOCHS):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        
        # Print epoch statistics
        if (epoch_number + 1) % 5 == 0:
            print(f"Epoch {epoch_number + 1:2d}: "
                  f"Loss {total_loss / len(train_set):1.4f}, "
                  f"LR {scheduler.get_last_lr()[0]:.4f}")
            
            # Print LRTT statistics for first layer
            if hasattr(model[0], 'analog_module') and hasattr(model[0].analog_module, 'controller'):
                controller = model[0].analog_module.controller
                print(f"  Layer 1 LRTT - A updates: {controller.num_a_updates}, "
                      f"B updates: {controller.num_b_updates}, "
                      f"Transfers: {controller.num_transfers}")

    print(f"\nTraining Time: {(time() - time_init) / 60:.2f} mins")


def test_evaluation(model, val_set):
    """Test the trained network

    Args:
        model (nn.Module): Trained model to be evaluated.
        val_set (DataLoader): Validation set to perform the evaluation.
    """
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)
            
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Print final LRTT statistics for all layers
    print("\nFinal LRTT Statistics:")
    for i, layer in enumerate(model):
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            controller = layer.analog_module.controller
            print(f"  Layer {i//2 + 1} (rank={controller.rank}): "
                  f"A updates: {controller.num_a_updates}, "
                  f"B updates: {controller.num_b_updates}, "
                  f"Transfers: {controller.num_transfers}")


def main():
    """Train a PyTorch analog model with LRTT to classify MNIST."""
    # Load datasets.
    train_data, validation_data = load_images()

    # Prepare the model with LRTT layers
    model = create_analog_network_lrtt(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Train the model
    print("\nStarting LRTT training on MNIST...")
    print("=" * 50)
    train(model, train_data)

    # Evaluate the trained model
    test_evaluation(model, validation_data)


if __name__ == "__main__":
    main()