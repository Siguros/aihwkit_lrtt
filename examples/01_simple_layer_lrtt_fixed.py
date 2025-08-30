# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 1 with LRTT: simple network with one layer using LRTT.

Simple network that consists of one analog layer with LRTT (Low-Rank Tensor-Train).
The network aims to learn to sum all the elements from one array.
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
import torch
from torch import Tensor
from torch.nn.functional import mse_loss

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.rpu_base import cuda

# Set seed for reproducibility
torch.manual_seed(42)

# Create more training data for better learning
# Original had only 2 samples, let's create more
def create_sum_data(n_samples=10):
    """Create data where output is related to input sum"""
    x_data = torch.rand(n_samples, 4) 
    # Output is based on weighted sum of inputs
    y_data = torch.zeros(n_samples, 2)
    for i in range(n_samples):
        sum_val = x_data[i].sum()
        y_data[i, 0] = sum_val * 1.5  # First output
        y_data[i, 1] = sum_val * 0.5  # Second output
    return x_data, y_data

# Create training data
x, y = create_sum_data(20)

# Configure LRTT with settings that work well
device_cfg = PythonLRTTPreset.idealized(
    rank=2,                    # Low-rank approximation rank
    transfer_every=100,         # Transfer A⊗B to C frequently
    lora_alpha=2.0            # Higher LoRA scaling for better learning
)
device_cfg.transfer_lr = device_cfg.lora_alpha
device_cfg.correct_gradient_magnitudes = True  # Important for convergence

# Create RPU configuration
rpu_config = PythonLRTTRPUConfig(device=device_cfg)

# Create the analog linear layer with LRTT (no bias for LRTT)
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Define an analog-aware optimizer
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

print("Training simple layer with LRTT (rank=2)...")
print("Task: Learn to output weighted sums of inputs")
print("-" * 50)

losses = []
for epoch in range(200):
    # Shuffle data each epoch
    if epoch % 50 == 0 and epoch > 0:
        perm = torch.randperm(x.size(0))
        x = x[perm]
        y = y[perm]
    
    # Forward pass
    pred = model(x)
    loss = mse_loss(pred, y)
    losses.append(loss.item())
    
    # Backward pass
    opt.zero_grad()
    loss.backward()
    
    # Update weights
    opt.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss:.8f}")

print("-" * 50)
print(f"Initial loss: {losses[0]:.8f}")
print(f"Final loss: {losses[-1]:.8f}")
print(f"Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")

# Check if training was successful
if losses[-1] < losses[0] * 0.5:
    print("✅ Training successful: loss decreased by >50%")
elif losses[-1] < losses[0] * 0.9:
    print("⚠️  Training partially successful: loss decreased by >10%")
else:
    print("❌ Training failed: loss didn't decrease significantly")

# Check LRTT statistics
if hasattr(model.analog_module, 'controller'):
    controller = model.analog_module.controller
    print(f"\nLRTT Statistics:")
    print(f"  A updates: {controller.num_a_updates}")
    print(f"  B updates: {controller.num_b_updates}")
    print(f"  Transfers: {controller.num_transfers}")

# Test on new data
print(f"\nTesting on new data:")
x_test, y_test = create_sum_data(5)
if cuda.is_compiled():
    x_test = x_test.cuda()
    y_test = y_test.cuda()

with torch.no_grad():
    pred_test = model(x_test)
    test_loss = mse_loss(pred_test, y_test)
    print(f"Test loss: {test_loss:.8f}")
    
    # Show a few examples
    print("\nExample predictions:")
    for i in range(min(3, x_test.size(0))):
        input_sum = x_test[i].sum().item()
        print(f"  Input sum: {input_sum:.3f}")
        print(f"    Expected: [{y_test[i,0]:.3f}, {y_test[i,1]:.3f}]")
        print(f"    Predicted: [{pred_test[i,0]:.3f}, {pred_test[i,1]:.3f}]")