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

# Prepare the datasets (input and expected output).
# Using the original data but repeated to have more samples
x_base = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y_base = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Repeat data to have more samples for better training
x = x_base.repeat(5, 1)  # Create 10 samples
y = y_base.repeat(5, 1)  # Create 10 samples

# Add small noise to make samples slightly different
torch.manual_seed(42)
x = x + torch.randn_like(x) * 0.01

# Configure LRTT with proper settings for convergence
device_config = PythonLRTTPreset.idealized(
    rank=2,                    # Low-rank approximation rank
    transfer_every=10,         # Transfer A⊗B to C every 10 updates
    lora_alpha=2.0            # LoRA scaling factor (higher works better)
)
device_config.transfer_lr = device_config.lora_alpha
device_config.correct_gradient_magnitudes = True  # Important for convergence

# Create RPU configuration
rpu_config = PythonLRTTRPUConfig(device=device_config)

# Create the analog linear layer with LRTT
# Note: LRTT doesn't support bias, so use bias=False
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

print("Training simple layer with LRTT (rank=2)...")
print("-" * 50)

losses = []
for epoch in range(100):
    # Delete old gradient
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()

    opt.step()
    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Loss error: {loss:.8f}")

print("-" * 50)
print(f"Initial loss: {losses[0]:.8f}")
print(f"Final loss: {losses[-1]:.8f}")
print(f"Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")

# Check LRTT statistics
if hasattr(model.analog_module, 'controller'):
    controller = model.analog_module.controller
    print(f"\nLRTT Statistics:")
    print(f"  A updates: {controller.num_a_updates}")
    print(f"  B updates: {controller.num_b_updates}")
    print(f"  Transfers: {controller.num_transfers}")

# Test on original data points
print("\nFinal predictions on original data:")
with torch.no_grad():
    # Test on the first two original samples
    x_test = x_base
    if cuda.is_compiled():
        x_test = x_test.cuda()
    pred_test = model(x_test)
    
    print(f"  Sample 1: input={x_base[0].tolist()}")
    print(f"           target={y_base[0].tolist()}")
    print(f"           predicted={pred_test[0].tolist()}")
    print(f"  Sample 2: input={x_base[1].tolist()}")
    print(f"           target={y_base[1].tolist()}")
    print(f"           predicted={pred_test[1].tolist()}")