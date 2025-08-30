# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 3 baseline: MNIST training using standard analog layers.

MNIST training example using standard analog layers (no LRTT) as baseline comparison.
Uses IdealizedPreset and TikitakaIdealizedPreset for comparison with LRTT performance.
"""
# pylint: disable=invalid-name, redefined-outer-name

import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import pandas as pd

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets.configs import IdealizedPreset, TikiTakaIdealizedPreset
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
EPOCHS = 50
BATCH_SIZE = 64

# Device configuration - choose one
DEVICE_TYPE = "TikiTakaIdealizedPreset"  # Options: "IdealizedPreset", "TikiTakaIdealizedPreset"


def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
    ])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def get_rpu_config():
    """Get RPU configuration based on DEVICE_TYPE.
    
    Returns:
        RPU configuration (IdealizedPreset or TikiTakaIdealizedPreset)
    """
    if DEVICE_TYPE == "IdealizedPreset":
        return IdealizedPreset()
    elif DEVICE_TYPE == "TikiTakaIdealizedPreset":
        return TikiTakaIdealizedPreset()
    else:
        raise ValueError(f"Unknown DEVICE_TYPE: {DEVICE_TYPE}")


def create_analog_network_baseline(input_size, hidden_sizes, output_size):
    """Create the neural network using standard analog layers (no LRTT).

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model (baseline)
    """
    print(f"Creating baseline network with {DEVICE_TYPE}")
    
    rpu_config = get_rpu_config()
    
    model = AnalogSequential(
        # Layer 1: 784 -> 256 
        AnalogLinear(
            input_size,
            hidden_sizes[0],
            bias=True,
            rpu_config=rpu_config,
        ),
        nn.ReLU(),
        # Layer 2: 256 -> 128 
        AnalogLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            bias=True,
            rpu_config=rpu_config,
        ),
        nn.ReLU(),
        # Layer 3: 128 -> 10 (final classification layer)
        AnalogLinear(
            hidden_sizes[1],
            output_size,
            bias=True,
            rpu_config=rpu_config,
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    print(model)
    
    # Print configuration details
    print(f"\nBaseline Configuration:")
    print(f"  Device: {DEVICE_TYPE}")
    print(f"  All layers use: {DEVICE_TYPE}")
    print(f"  No LRTT decomposition")
    print(f"  Standard analog layers with bias enabled\n")
    
    return model


def create_sgd_optimizer(model):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=0.1)  # Same LR as LRTT version
    optimizer.regroup_param_groups(model)

    return optimizer


def train(model, train_set, val_set):
    """Train the network (single phase, no warmup).

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
        val_set (DataLoader): dataset of elements to use for validation.
    """
    classifier = nn.NLLLoss()
    optimizer = create_sgd_optimizer(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Results tracking
    results = []
    
    print("Single phase baseline training")

    time_init = time()
    for epoch_number in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_set):
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
            
            # Track accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(labels.view_as(pred)).sum().item()
            total_samples += images.size(0)

            # Basic diagnostics every 200 batches (less frequent than LRTT version)
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"[Epoch {epoch_number:02d}, Batch {batch_idx:04d}] "
                      f"Loss={loss.item():.6f}")

        # End of epoch statistics
        epoch_accuracy = 100. * correct_predictions / total_samples
        avg_loss = total_loss / len(train_set)
        
        # Validation (standard validation, no C-only needed)
        val_accuracy = validate(model, val_set)
        
        scheduler.step()
        
        print(f"Epoch {epoch_number + 1:2d}: "
              f"Loss={avg_loss:.4f}, "
              f"Train_Acc={epoch_accuracy:.2f}%, "
              f"Val_Acc={val_accuracy:.2f}%, "
              f"LR={scheduler.get_last_lr()[0]:.4f}")
        
        # Store results
        results.append({
            'epoch': epoch_number + 1,
            'train_loss': avg_loss,
            'train_accuracy': epoch_accuracy,
            'val_accuracy': val_accuracy,
            'learning_rate': scheduler.get_last_lr()[0]
        })

    print(f"\nTraining Time: {(time() - time_init) / 60:.2f} mins")
    
    return results


def validate(model, val_set):
    """Standard validation.
    
    Args:
        model (nn.Module): Trained model to be evaluated.
        val_set (DataLoader): Validation set.
        
    Returns:
        float: Validation accuracy.
    """
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)
            
            # Standard forward pass
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.train()
    return 100. * correct / total


def test_evaluation(model, val_set):
    """Test the trained network

    Args:
        model (nn.Module): Trained model to be evaluated.
        val_set (DataLoader): Validation set to perform the evaluation.
        
    Returns:
        float: Test accuracy percentage.
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
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
    
    model.train()
    return accuracy


def save_results_to_excel(results, final_test_accuracy):
    """Save training results to Excel file with parameter-based filename."""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename based on device type
    filename = f"mnist_baseline_results_{DEVICE_TYPE}_e{EPOCHS}.xlsx"
    filepath = os.path.join(results_dir, filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Add final test accuracy to last row
    df.loc[len(df)-1, 'test_accuracy'] = final_test_accuracy
    
    # Add parameter information
    params_info = {
        'parameter': ['device_type', 'epochs', 'batch_size', 'input_size', 'hidden_layer1', 'hidden_layer2', 'output_size'],
        'value': [DEVICE_TYPE, EPOCHS, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZES[0], HIDDEN_SIZES[1], OUTPUT_SIZE]
    }
    params_df = pd.DataFrame(params_info)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Training_Results', index=False)
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    """Train a PyTorch analog model (baseline) to classify MNIST."""
    # Load datasets.
    train_data, validation_data = load_images()

    # Prepare the model with standard analog layers
    model = create_analog_network_baseline(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Train the model
    print("\nStarting baseline training on MNIST...")
    print("=" * 50)
    results = train(model, train_data, validation_data)

    # Evaluate the trained model
    final_test_accuracy = test_evaluation(model, validation_data)
    
    # Save results to Excel
    save_results_to_excel(results, final_test_accuracy)


if __name__ == "__main__":
    main()