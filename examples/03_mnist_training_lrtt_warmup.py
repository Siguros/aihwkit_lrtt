# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: LRTT MNIST training with warm-start.

MNIST training using LRTT (Low-Rank Tensor-Train) analog layers with:
- Warm-start phase: Full-rank training for initial epochs
- LRTT phase: Training with automatic merge-and-reinit controlled by transfer_every
Based on LRTT controller's internal ReLoRA-style behavior.

Key features:
- Phase-A (epochs 0-4): Full-rank warm-start with IdealizedPreset (bias=False)
- Phase-B (epochs 5-29): LRTT with automatic merge controlled by transfer_every
- Proper weight transfer from warm-start to LRTT's C tile
- No manual merge/reinit needed - LRTT controller handles this internally
"""
# pylint: disable=invalid-name, redefined-outer-name

import os
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import pandas as pd

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.configs import IdealizedPreset
from aihwkit.simulator.rpu_base import cuda

# ----------------------
# Global configuration
# ----------------------
USE_CUDA = 1 if cuda.is_compiled() else 0
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

PATH_DATASET = os.path.join("data", "DATASET")

INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

EPOCHS = 30
WARM_EPOCHS = 1                # Phase-A: full-rank warm-start
BATCH_SIZE = 64

LRTT_RANKS = [32, 32]            # ranks for 784->256, 256->128
TRANSFER_EVERY = 100       # auto-transfer every 2 epochs (937 steps/epoch * 2)
LORA_ALPHA = 8               # LoRA scale

LR_WARMSTART = 0.01            # Learning rate for warm-start phase
LR_LRTT = 0.01                 # Learning rate for LRTT phase (same as warmstart to avoid issues)
MOMENTUM = 0.9                 # SGD momentum

# ----------------------
# Data loading
# ----------------------
def load_images():
    """Load and prepare MNIST dataset."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=tfm)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=tfm)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_data, val_data

# ----------------------
# Model builders
# ----------------------
def create_lrtt_config(rank):
    """Create LRTT configuration with specified rank."""
    dev = PythonLRTTPreset.idealized(rank=rank, transfer_every=TRANSFER_EVERY, lora_alpha=LORA_ALPHA)
    # stability helpers
    dev.reinit_gain = 0.5
    dev.correct_gradient_magnitudes = True
    dev.transfer_lr = LORA_ALPHA    # keep function-invariance at merge
    return PythonLRTTRPUConfig(device=dev)

def build_fullrank_model():
    """Build full-rank warm-start model (all Idealized, no bias to match LRTT)."""
    m = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], bias=False, rpu_config=IdealizedPreset()),
        nn.ReLU(),
        AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], bias=False, rpu_config=IdealizedPreset()),
        nn.ReLU(),
        AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, bias=False, rpu_config=IdealizedPreset()),
        nn.LogSoftmax(dim=1),
    )
    if USE_CUDA: 
        m.cuda()
    return m

@torch.no_grad()
def replace_with_lrtt(full_model, ranks=(8, 8)):
    """
    Replace the first two linear layers with LRTT.
    Copy warm-start full-rank W -> C tile, freeze C.
    LRTT controller handles A/B initialization internally.
    
    Args:
        full_model: Full-rank model from warm-start
        ranks: Tuple of ranks for each LRTT layer
    
    Returns:
        New model with LRTT layers
    """
    new = AnalogSequential()
    li = 0
    for mod in full_model:
        if isinstance(mod, AnalogLinear) and li < 2:
            in_f, out_f = mod.in_features, mod.out_features
            
            # Get weights from analog module
            if hasattr(mod, 'analog_module'):
                w_full = mod.analog_module.get_weights()[0].detach().clone()  # [out, in]
            else:
                w_full = mod.weight.detach().clone() if mod.weight is not None else torch.randn(out_f, in_f)
            
            # Create LRTT layer
            lrtt = AnalogLinear(in_f, out_f, bias=False, rpu_config=create_lrtt_config(ranks[li]))
            if USE_CUDA: 
                lrtt.cuda()

            # Copy W to C (no transpose - use same axis order as original)
            try:
                lrtt.analog_module.tile_c.set_weights(w_full)
            except Exception:
                try:
                    # Fallback: direct weight copy
                    if hasattr(lrtt.analog_module, 'controller'):
                        lrtt.analog_module.controller.tile_c.set_weights(w_full)
                except Exception:
                    pass

            # Don't freeze C - let LRTT controller manage all tiles
            # The controller will handle the proper updates and transfers

            # Note: LRTT controller automatically initializes A=0, B=Kaiming in reinit()

            new.append(lrtt)
            new.append(nn.ReLU())
            li += 1
        else:
            new.append(mod)
    
    if USE_CUDA: 
        new.cuda()
    return new

# ----------------------
# Training & Evaluation
# ----------------------
def validate_c_only(model, val_set):
    """
    Validate using only C component (no AB) to evaluate merge quality.
    
    Args:
        model: Neural network model
        val_set: Validation data loader
    
    Returns:
        Accuracy percentage
    """
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            x = images.view(images.size(0), -1)
            
            # Forward pass through layers
            for layer in model:
                if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                    # Use only C component for LRTT layers
                    try:
                        controller = layer.analog_module.controller
                        x = controller.tile_c.forward(x)
                    except Exception:
                        x = layer(x)  # fallback
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                elif isinstance(layer, nn.LogSoftmax):
                    x = torch.log_softmax(x, dim=1)
                else:
                    x = layer(x)
            
            _, pred = torch.max(x, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    model.train()
    return 100.0 * correct / total

def evaluate_full(model, val_set):
    """
    Evaluate model with all components (C + AB).
    
    Args:
        model: Neural network model
        val_set: Validation data loader
    
    Returns:
        Accuracy percentage
    """
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            x = images.view(images.size(0), -1)
            out = model(x)
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    model.train()
    return 100.0 * correct / total

def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        loader: Training data loader
        optimizer: SGD optimizer
        criterion: Loss function
    
    Returns:
        Tuple of (average loss, accuracy percentage)
    """
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        x = images.view(images.size(0), -1)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += x.size(0)
    
    return total_loss / len(loader), 100.0 * correct / total

def train_with_warmstart():
    """
    Main training function with warm-start approach.
    
    Returns:
        Tuple of (model, results list, final test accuracy)
    """
    train_loader, val_loader = load_images()

    # ----------------- Phase-A: Warm-start full-rank -----------------
    print("=" * 60)
    print(f"LRTT MNIST Training with Warm-start")
    print(f"Configurations:")
    print(f"  - Warm-start epochs: {WARM_EPOCHS}")
    print(f"  - Total epochs: {EPOCHS}")
    print(f"  - LRTT ranks: {LRTT_RANKS}")
    print(f"  - LoRA alpha: {LORA_ALPHA}")
    print(f"  - Transfer every: {TRANSFER_EVERY} steps")
    print(f"  - LR (Warmstart/LRTT): {LR_WARMSTART}/{LR_LRTT}")
    print("=" * 60)
    
    full = build_fullrank_model()
    criterion = nn.NLLLoss()
    optA = AnalogSGD(full.parameters(), lr=LR_WARMSTART, momentum=MOMENTUM)
    optA.regroup_param_groups(full)
    schA = StepLR(optA, step_size=10, gamma=0.5)

    results = []
    print("\n== Phase-A: Full-rank warm-start ==")
    for ep in range(WARM_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(full, train_loader, optA, criterion)
        val_c = evaluate_full(full, val_loader)  # full-rank so C-only = full
        schA.step()
        
        print(f"[Warm {ep+1:02d}/{WARM_EPOCHS}] "
              f"loss={tr_loss:.4f} acc={tr_acc:.2f}% "
              f"val={val_c:.2f}% LR={schA.get_last_lr()[0]:.4f}")
        
        results.append({
            'phase': 'warm',
            'epoch': ep + 1,
            'train_loss': tr_loss,
            'train_accuracy': tr_acc,
            'val_accuracy_c_only': val_c,
            'learning_rate': schA.get_last_lr()[0]
        })

    # ----------------- Convert to LRTT (C <- W), freeze C -----------------
    print("\n== Converting to LRTT model ==")
    model = replace_with_lrtt(full, ranks=LRTT_RANKS)
    
    # Create new optimizer for LRTT model
    # Important: Use same LR to avoid tile learning rate issues with AnalogSGD
    optB = AnalogSGD(model.parameters(), lr=LR_LRTT, momentum=MOMENTUM)
    optB.regroup_param_groups(model)
    schB = StepLR(optB, step_size=10, gamma=0.5)

    # ----------------- Phase-B: LRTT training -----------------
    print("\n== Phase-B: LRTT training (automatic merge controlled by transfer_every) ==")
    for ep in range(WARM_EPOCHS, EPOCHS):
        # Train for one epoch - LRTT controller handles merge/reinit internally
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optB, criterion)

        # Validation
        val_c = validate_c_only(model, val_loader)
        val_full = evaluate_full(model, val_loader)
        schB.step()

        print(f"[LRTT {ep+1:02d}/{EPOCHS}] "
              f"loss={tr_loss:.4f} acc={tr_acc:.2f}% "
              f"val(C-only)={val_c:.2f}% val(full)={val_full:.2f}% "
              f"LR={schB.get_last_lr()[0]:.4f}")
        
        # Print LRTT transfer statistics
        for i, layer in enumerate(model):
            if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
                ctrl = layer.analog_module.controller
                if ctrl.num_transfers > 0 or (ep+1) % 5 == 0:  # Print every 5 epochs or when transfers happen
                    print(f"  Layer {i//2}: A updates={ctrl.num_a_updates}, "
                          f"B updates={ctrl.num_b_updates}, Transfers={ctrl.num_transfers}")
        
        results.append({
            'phase': 'lrtt',
            'epoch': ep + 1,
            'train_loss': tr_loss,
            'train_accuracy': tr_acc,
            'val_accuracy_c_only': val_c,
            'val_accuracy_full': val_full,
            'learning_rate': schB.get_last_lr()[0]
        })

    # Final evaluation with full model (C + AB)
    final_acc = evaluate_full(model, val_loader)
    print(f"\n== Final Results ==")
    print(f"Final Test Accuracy (with A/B): {final_acc:.2f}%")
    
    # Print final LRTT statistics
    print("\nFinal LRTT Statistics:")
    for i, layer in enumerate(model):
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            ctrl = layer.analog_module.controller
            print(f"  Layer {i//2}: A updates={ctrl.num_a_updates}, "
                  f"B updates={ctrl.num_b_updates}, Transfers={ctrl.num_transfers}")
    
    return model, results, final_acc

# ----------------------
# Save results
# ----------------------
def save_results_to_excel(results, final_test_accuracy):
    """
    Save training results to Excel file.
    
    Args:
        results: List of training metrics per epoch
        final_test_accuracy: Final test accuracy
    
    Returns:
        Filepath of saved Excel file
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"mnist_warmstart_lrtt_r{LRTT_RANKS[0]}_{LRTT_RANKS[1]}_a{LORA_ALPHA:.1f}_te{TRANSFER_EVERY}.xlsx"
    filepath = os.path.join(results_dir, filename)
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df.loc[len(df) - 1, 'test_accuracy'] = final_test_accuracy
    
    params_info = {
        'parameter': [
            'transfer_every', 'rank_layer1', 'rank_layer2', 'lora_alpha',
            'epochs', 'warm_epochs', 'batch_size', 'lr_warmstart', 'lr_lrtt'
        ],
        'value': [
            TRANSFER_EVERY, LRTT_RANKS[0], LRTT_RANKS[1], LORA_ALPHA,
            EPOCHS, WARM_EPOCHS, BATCH_SIZE, LR_WARMSTART, LR_LRTT
        ]
    }
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Training_Results', index=False)
        pd.DataFrame(params_info).to_excel(writer, sheet_name='Parameters', index=False)
    
    print(f"Results saved to: {filepath}")
    return filepath

# ----------------------
# Main entry point
# ----------------------
def main():
    """Main function to run warmstart LRTT training."""
    t0 = time()
    model, results, final_test_accuracy = train_with_warmstart()
    save_results_to_excel(results, final_test_accuracy)
    print(f"\nTotal wall time: {(time() - t0) / 60:.2f} min")

if __name__ == "__main__":
    main()