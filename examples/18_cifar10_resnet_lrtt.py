# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with LRTT: ResNet32 CNN with CIFAR10 using LRTT layers.

CIFAR10 dataset on a ResNet inspired network using LRTT (Low-Rank Tensor-Train)
analog layers based on the paper: https://arxiv.org/abs/1512.03385
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime

# Imports from PyTorch.
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_LRTT")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "example_18_lrtt_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 100  # Reduced for LRTT demonstration
BATCH_SIZE = 32
LEARNING_RATE = 0.1
N_CLASSES = 10

# LRTT configuration parameters
LRTT_RANK_CONV = 8  # Rank for convolutional layers
LRTT_RANK_FC = 16  # Rank for fully connected layers
TRANSFER_EVERY = 200  # Transfer A⊗B to C every N updates
LORA_ALPHA = 1.0  # LoRA scaling factor


def create_lrtt_config_conv():
    """Create LRTT configuration for convolutional layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for conv layers
    """
    device_config = PythonLRTTPreset.idealized(
        rank=LRTT_RANK_CONV,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        forward_inject=True,
        correct_gradient_magnitudes=True
    )
    device_config.transfer_lr = device_config.lora_alpha
    
    # Add mapping for larger layers
    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        max_input_size=512,
        max_output_size=512
    )
    
    return PythonLRTTRPUConfig(device=device_config, mapping=mapping)


def create_lrtt_config_fc():
    """Create LRTT configuration for fully connected layers.
    
    Returns:
        PythonLRTTRPUConfig: LRTT configuration for FC layers
    """
    device_config = PythonLRTTPreset.idealized(
        rank=LRTT_RANK_FC,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        forward_inject=True,
        correct_gradient_magnitudes=True
    )
    device_config.transfer_lr = device_config.lora_alpha
    
    return PythonLRTTRPUConfig(device=device_config)


class ResidualBlockLRTT(nn.Module):
    """Residual block with LRTT analog convolutional layers."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        # Use LRTT analog convolutional layers
        self.conv1 = AnalogConv2d(
            in_ch, hidden_ch, 
            kernel_size=3, padding=1, stride=stride,
            bias=False,  # LRTT doesn't support bias
            rpu_config=create_lrtt_config_conv()
        )
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        
        self.conv2 = AnalogConv2d(
            hidden_ch, hidden_ch,
            kernel_size=3, padding=1,
            bias=False,
            rpu_config=create_lrtt_config_conv()
        )
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = AnalogConv2d(
                in_ch, hidden_ch,
                kernel_size=1, stride=stride,
                bias=False,
                rpu_config=create_lrtt_config_conv()
            )
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks_lrtt(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple LRTT residual blocks to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlockLRTT(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlockLRTT(hidden_ch, hidden_ch))
    return layers


def create_model():
    """ResNet34 inspired analog model with LRTT layers.

    Returns:
       nn.Module: created model with LRTT
    """

    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

    # Input layer uses FloatingPointDevice for better stability
    l0 = nn.Sequential(
        AnalogConv2d(
            3, channel[0],
            kernel_size=3, stride=1, padding=1,
            bias=True,  # Can use bias with FloatingPoint
            rpu_config=FloatingPointRPUConfig()
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    l1 = nn.Sequential(
        *concatenate_layer_blocks_lrtt(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks_lrtt(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks_lrtt(channel[1], channel[2], block_per_layers[2]))
    
    # Final classification layer uses FloatingPointDevice for better stability
    from aihwkit.nn import AnalogLinear
    l4 = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        AnalogLinear(
            channel[2], N_CLASSES,
            bias=True,  # Can use bias with FloatingPoint
            rpu_config=FloatingPointRPUConfig()
        )
    )

    model = nn.Sequential(l0, l1, l2, l3, l4)
    
    print(f"\nCreated ResNet with LRTT layers:")
    print(f"  Input layer: FloatingPointDevice")
    print(f"  Conv layers rank: {LRTT_RANK_CONV} (LRTT)")
    print(f"  Final FC layer: FloatingPointDevice")
    print(f"  Transfer every: {TRANSFER_EVERY} updates")
    print(f"  LoRA alpha: {LORA_ALPHA}\n")
    
    return model


def load_images():
    """Load images for train from torchvision datasets.

    Returns:
        Dataset, Dataset: train data and validation data"""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.CIFAR10(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, validation_data


def create_sgd_optimizer(model, learning_rate):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained
        learning_rate (float): global parameter to define learning rate

    Returns:
        Optimizer: created analog optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    return optimizer


def train_step(train_data, model, criterion, optimizer):
    """Train network.

    Args:
        train_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss
        optimizer (Optimizer): analog model optimizer

    Returns:
        nn.Module, Optimizer, float: model, optimizer, and epoch loss
    """
    total_loss = 0

    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss


def test_evaluation(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float: model, test epoch loss, test error, and test accuracy
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    with no_grad():
        for images, labels in validation_data:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch_max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
        epoch_loss = total_loss / len(validation_data.dataset)
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    return model, epoch_loss, error, accuracy


def print_lrtt_statistics(model, epoch):
    """Print LRTT statistics for monitoring.
    
    Args:
        model (nn.Module): Model with LRTT layers
        epoch (int): Current epoch number
    """
    if epoch % 10 == 0:  # Print every 10 epochs
        print(f"\nLRTT Statistics at epoch {epoch}:")
        
        # Count LRTT layers and get statistics
        lrtt_count = 0
        total_transfers = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
                controller = module.analog_module.controller
                lrtt_count += 1
                total_transfers += controller.num_transfers
                
                if lrtt_count <= 3:  # Print first 3 layers
                    print(f"  {name}: A/B updates={controller.num_a_updates}, "
                          f"Transfers={controller.num_transfers}")
        
        print(f"  Total LRTT layers: {lrtt_count}")
        print(f"  Total transfers: {total_transfers}\n")


def main():
    """Train a PyTorch ResNet analog model with LRTT to classify CIFAR10."""
    # Seed
    manual_seed(SEED)

    # Load the images.
    train_data, validation_data = load_images()

    # Make the model
    model = create_model()

    if USE_CUDA:
        model = model.to(DEVICE)

    print(f"Model moved to {DEVICE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_sgd_optimizer(model, LEARNING_RATE)

    best_accuracy = 0
    best_epoch = 0
    
    print("\nStarting LRTT training on CIFAR10...")
    print("=" * 60)

    for epoch in range(N_EPOCHS):
        # Train
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)

        # Validate
        model, val_loss, error, accuracy = test_evaluation(validation_data, model, criterion)

        # Update learning rate
        if epoch == 30 or epoch == 60:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.1
                print(f"Reducing learning rate to {param_group['lr']}")

        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            save(model.state_dict(), WEIGHT_PATH)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss {train_loss:.4f}, "
                  f"Val Loss {val_loss:.4f}, "
                  f"Val Acc {accuracy:.2f}%, "
                  f"Val Err {error:.2f}%")
        
        # Print LRTT statistics
        print_lrtt_statistics(model, epoch + 1)

    print("=" * 60)
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}% at epoch {best_epoch + 1}")
    print(f"Model weights saved to: {WEIGHT_PATH}")


if __name__ == "__main__":
    main()