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
import pandas as pd

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.configs import IdealizedPreset
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

# LRTT parameters - using different ranks for different layer sizes
LRTT_RANKS = [32, 32]  # Ranks for LRTT layers (input->hidden1, hidden1->hidden2)
TRANSFER_EVERY = 1000 #fer A⊗B to C every N updates
LORA_ALPHA = 32 #LoRA scaling factor - reduced from 100.0 for stability


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
    
    # Stability improvements
    device_config.reinit_gain = 0.5  # Increased from 0.1 for stronger initial signals
    device_config.correct_gradient_magnitudes = True  # Important for convergence
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
    print(f"Creating LRTT network with ranks: {LRTT_RANKS} (final layer uses IdealizedPreset)")
    
    model = AnalogSequential(
        # Layer 1: 784 -> 256 with rank 32
        AnalogLinear(
            input_size,
            hidden_sizes[0],
            bias=False,  # LRTT doesn't support bias
            rpu_config=create_lrtt_config(LRTT_RANKS[0]),
        ),
        nn.ReLU(),  # Changed from Sigmoid to avoid saturation
        # Layer 2: 256 -> 128 with rank 16
        AnalogLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            bias=False,
            rpu_config=create_lrtt_config(LRTT_RANKS[1]),
        ),
        nn.ReLU(),  # Changed from Sigmoid to avoid saturation
        # Layer 3: 128 -> 10 with IdealizedPresetDevice (final classification layer)
        AnalogLinear(
            hidden_sizes[1],
            output_size,
            bias=False,  # Can use bias with IdealizedPreset
            rpu_config=IdealizedPreset(),
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    print(model)
    
    # Print LRTT configuration details
    print("\nLRTT Configuration:")
    print(f"  LRTT Ranks: {LRTT_RANKS} (layers 1-2)")
    print(f"  Final layer: IdealizedPresetDevice")
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
    optimizer = AnalogSGD(model.parameters(), lr=0.01)  # Reduced from 0.05 for stability
    optimizer.regroup_param_groups(model)

    return optimizer


def train(model, train_set, val_set):
    """Train the network with single phase (no warmup).

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
    
    print("Single phase training - no warmup")

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

            # Comprehensive diagnostics every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                with torch.no_grad():
                    # Analyze first LRTT layer (layer 0)
                    if hasattr(model[0], 'analog_module'):
                        lrtt_layer = model[0].analog_module
                        controller = lrtt_layer.controller
                        
                        # Get component weights
                        try:
                            # Get current batch for signal analysis
                            x_flat = images[:8]  # Use first 8 samples for analysis
                            
                            # Forward through each component
                            Bx = lrtt_layer.tile_b.forward(x_flat)  # [8, rank]
                            ABx = lrtt_layer.tile_a.forward(Bx)     # [8, hidden]  
                            Cx = lrtt_layer.tile_c.forward(x_flat)  # [8, hidden]
                            y_combined = Cx + controller.lora_alpha * ABx
                            
                            # Signal magnitudes
                            bx_norm = Bx.norm().item()
                            abx_norm = ABx.norm().item() 
                            cx_norm = Cx.norm().item()
                            
                            # Component weight norms
                            try:
                                # Get weight tensors directly from tiles
                                A_weights = lrtt_layer.tile_a.get_weights()[0]
                                B_weights = lrtt_layer.tile_b.get_weights()[0] 
                                C_weights = lrtt_layer.tile_c.get_weights()[0]
                                
                                a_norm = A_weights.norm().item()
                                b_norm = B_weights.norm().item()
                                c_norm = C_weights.norm().item()
                                ab_norm = (A_weights @ B_weights).norm().item()
                            except:
                                a_norm = b_norm = c_norm = ab_norm = 0.0
                            
                            # Logits analysis (before LogSoftmax)
                            pre_softmax = model[:-1](images[:8])  # Everything except LogSoftmax
                            logit_var = pre_softmax.var().item()
                            
                            # Top-2 margin for classification quality
                            probs = torch.exp(output[:8])  # Convert log-softmax back to probabilities
                            top2_vals = torch.topk(probs, k=2, dim=1).values
                            margin_mean = (top2_vals[:,0] - top2_vals[:,1]).mean().item()
                            
                            print(f"[Epoch {epoch_number:02d}, Batch {batch_idx:04d}] "
                                  f"‖Bx‖={bx_norm:.3f} ‖ABx‖={abx_norm:.3f} ‖Cx‖={cx_norm:.3f} "
                                  f"logit_var={logit_var:.3f} margin={margin_mean:.3f}")
                            print(f"  Weights: ‖A‖={a_norm:.3e} ‖B‖={b_norm:.3e} ‖C‖={c_norm:.3e} ‖AB‖={ab_norm:.3e}")
                            print(f"  Updates: A={controller.num_a_updates} B={controller.num_b_updates} T={controller.num_transfers}")
                            
                        except Exception as e:
                            print(f"[Batch {batch_idx}] Diagnostic error: {e}")

        # End of epoch statistics
        epoch_accuracy = 100. * correct_predictions / total_samples
        avg_loss = total_loss / len(train_set)
        
        # Validation with C-only (no A/B contribution)
        val_accuracy_c_only = validate_c_only(model, val_set)
        
        scheduler.step()
        
        print(f"Epoch {epoch_number + 1:2d}: "
              f"Loss={avg_loss:.4f}, "
              f"Train_Acc={epoch_accuracy:.2f}%, "
              f"Val_Acc_C_only={val_accuracy_c_only:.2f}%, "
              f"LR={scheduler.get_last_lr()[0]:.4f}")
        
        # Store results
        results.append({
            'epoch': epoch_number + 1,
            'train_loss': avg_loss,
            'train_accuracy': epoch_accuracy,
            'val_accuracy_c_only': val_accuracy_c_only,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # Print LRTT statistics for first layer only
        if hasattr(model[0], 'analog_module') and hasattr(model[0].analog_module, 'controller'):
            controller = model[0].analog_module.controller
            print(f"  Layer 0 LRTT: A={controller.num_a_updates}, "
                  f"B={controller.num_b_updates}, Transfers={controller.num_transfers}")

    print(f"\nTraining Time: {(time() - time_init) / 60:.2f} mins")
    
    return results


def validate_c_only(model, val_set):
    """Validate using only C tiles (no A/B contribution).
    
    Args:
        model (nn.Module): Trained model to be evaluated.
        val_set (DataLoader): Validation set.
        
    Returns:
        float: Validation accuracy using C-only forward pass.
    """
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            images = images.view(images.shape[0], -1)
            
            # Forward pass using only C tiles
            x = images
            for i, layer in enumerate(model):
                if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                    # LRTT layer: use only C tile
                    controller = layer.analog_module.controller
                    x = controller.tile_c.forward(x)
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                elif isinstance(layer, nn.LogSoftmax):
                    x = torch.log_softmax(x, dim=1)
                elif hasattr(layer, 'analog_module'):
                    # Non-LRTT analog layer (final layer)
                    x = layer(x)
            
            _, predicted = torch.max(x.data, 1)
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
    print(f"\nFinal Test Accuracy (with A/B): {accuracy:.2f}%")
    
    # Print final LRTT statistics for all layers
    print("\nFinal LRTT Statistics:")
    for i, layer in enumerate(model):
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            controller = layer.analog_module.controller
            print(f"  Layer {i//2 + 1} (rank={controller.rank}): "
                  f"A updates: {controller.num_a_updates}, "
                  f"B updates: {controller.num_b_updates}, "
                  f"Transfers: {controller.num_transfers}")
    
    model.train()
    return accuracy


def save_results_to_excel(results, final_test_accuracy):
    """Save training results to Excel file with parameter-based filename."""
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename based on parameters
    filename = f"mnist_lrtt_results_te{TRANSFER_EVERY}_r{LRTT_RANKS[0]}_{LRTT_RANKS[1]}_a{LORA_ALPHA:.1f}.xlsx"
    filepath = os.path.join(results_dir, filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Add final test accuracy to last row
    df.loc[len(df)-1, 'test_accuracy'] = final_test_accuracy
    
    # Add parameter information
    params_info = {
        'parameter': ['transfer_every', 'rank_layer1', 'rank_layer2', 'lora_alpha', 'epochs', 'batch_size'],
        'value': [TRANSFER_EVERY, LRTT_RANKS[0], LRTT_RANKS[1], LORA_ALPHA, EPOCHS, BATCH_SIZE]
    }
    params_df = pd.DataFrame(params_info)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Training_Results', index=False)
        params_df.to_excel(writer, sheet_name='Parameters', index=False)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    """Train a PyTorch analog model with LRTT to classify MNIST."""
    # Load datasets.
    train_data, validation_data = load_images()

    # Prepare the model with LRTT layers
    model = create_analog_network_lrtt(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Train the model
    print("\nStarting LRTT training on MNIST...")
    print("=" * 50)
    results = train(model, train_data, validation_data)

    # Evaluate the trained model (full model with A/B)
    final_test_accuracy = test_evaluation(model, validation_data)
    
    # Save results to Excel
    save_results_to_excel(results, final_test_accuracy)


if __name__ == "__main__":
    main()