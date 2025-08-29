#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Training example using Python-level LRTT (Low-Rank Transfer Tiki-Taka).

This example demonstrates how to use the Python-level LRTT implementation
for training neural networks with analog hardware acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    PythonLRTTRPUConfig,
    lrtt_idealized_config,
    lrtt_constant_step_config,
    PythonLRTTDevice
)
from aihwkit.simulator.configs.devices import ConstantStepDevice


class LRTTNet(nn.Module):
    """Simple network using LRTT analog layers."""
    
    def __init__(self, lrtt_config):
        super().__init__()
        
        # Create analog layers with LRTT configuration
        self.fc1 = AnalogLinear(784, 256, bias=True, rpu_config=lrtt_config)
        self.fc2 = AnalogLinear(256, 128, bias=True, rpu_config=lrtt_config)
        self.fc3 = AnalogLinear(128, 10, bias=True, rpu_config=lrtt_config)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def create_lrtt_config(preset='idealized', rank=4, transfer_every=100):
    """Create LRTT configuration based on preset.
    
    Args:
        preset: Configuration preset ('idealized', 'constant_step', 'custom')
        rank: LoRA rank for decomposition
        transfer_every: Transfer frequency (steps)
        
    Returns:
        PythonLRTTRPUConfig instance
    """
    if preset == 'idealized':
        # Idealized configuration with minimal noise
        config = lrtt_idealized_config(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0
        )
    elif preset == 'constant_step':
        # Realistic constant step device
        config = lrtt_constant_step_config(
            rank=rank,
            transfer_every=transfer_every,
            dw_min=0.001
        )
    else:
        # Custom configuration with specific devices
        device = PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            transfer_lr=1.0,
            lora_alpha=1.0,
            reinit_gain=0.1,
            forward_inject=True,
            correct_gradient_magnitudes=False,
            unit_cell_devices=[
                ConstantStepDevice(dw_min=0.001, w_min=-1.0, w_max=1.0),  # A
                ConstantStepDevice(dw_min=0.001, w_min=-1.0, w_max=1.0),  # B
                ConstantStepDevice(dw_min=0.001, w_min=-1.0, w_max=1.0),  # Visible
            ]
        )
        config = PythonLRTTRPUConfig(device=device)
    
    return config


def train(model, device, train_loader, optimizer, epoch):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} '
          f'({accuracy:.2f}%)\n')


def test(model, device, test_loader):
    """Test the model."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    
    return accuracy


def main():
    """Main training function."""
    # Training settings
    batch_size = 64
    epochs = 3
    learning_rate = 0.01
    
    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create LRTT configuration
    print("\n=== LRTT Configuration ===")
    lrtt_config = create_lrtt_config(
        preset='constant_step',  # Try 'idealized' or 'custom' too
        rank=8,                   # LoRA rank
        transfer_every=100        # Transfer every 100 steps
    )
    print(f"Config: {lrtt_config.get_brief_info()}")
    print(f"Rank: {lrtt_config.device.rank}")
    print(f"Transfer every: {lrtt_config.device.transfer_every} steps")
    print(f"LoRA alpha: {lrtt_config.device.lora_alpha}")
    print(f"Forward injection: {lrtt_config.device.forward_inject}")
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with LRTT
    print("\n=== Creating LRTT Model ===")
    model = LRTTNet(lrtt_config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Analog optimizer
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n=== Starting Training ===")
    best_accuracy = 0
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.2f}%")
    
    print(f"\n=== Training Complete ===")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # Show LRTT statistics (if available)
    print("\n=== LRTT Layer Statistics ===")
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            tile = module.analog_tile
            if hasattr(tile, 'controller'):
                controller = tile.controller
                print(f"{name}:")
                print(f"  Transfers performed: {controller.num_transfers}")
                print(f"  A updates: {controller.num_a_updates}")
                print(f"  B updates: {controller.num_b_updates}")


if __name__ == '__main__':
    main()