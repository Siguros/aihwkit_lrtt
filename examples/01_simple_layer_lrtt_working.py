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
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Use the EXACT same configuration as test_analog_only_lrtt.py which works
torch.manual_seed(42)  # For reproducibility

device_cfg = PythonLRTTPreset.idealized(rank=2, transfer_every=10, lora_alpha=2.0)
device_cfg.transfer_lr = device_cfg.lora_alpha
device_cfg.correct_gradient_magnitudes = True

rpu_config = PythonLRTTRPUConfig(device=device_cfg)

# Create model WITHOUT bias (as in working example)
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Use AnalogSGD with same learning rate as working example
optimizer = AnalogSGD(model.parameters(), lr=0.1)

print("Training simple layer with LRTT (same config as working test)...")
print("-" * 50)

losses = []
for epoch in range(100):
    # Forward
    pred = model(x)
    loss = mse_loss(pred, y)
    losses.append(loss.item())
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update (via tile.update)
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss:.8f}")

print("-" * 50)
print(f"Initial loss: {losses[0]:.8f}")
print(f"Final loss: {losses[-1]:.8f}")
print(f"Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")

# Check if loss decreased
if losses[-1] < losses[0] * 0.9:
    print("✅ Training successful: loss decreased by >10%")
else:
    print("⚠️  Training may not be working: loss didn't decrease much")

# Check LRTT statistics
if hasattr(model.analog_module, 'controller'):
    controller = model.analog_module.controller
    print(f"\nLRTT Statistics:")
    print(f"  A updates: {controller.num_a_updates}")
    print(f"  B updates: {controller.num_b_updates}")
    print(f"  Transfers: {controller.num_transfers}")
    
# Verify the model learned something
print(f"\nPredictions vs Targets:")
with torch.no_grad():
    final_pred = model(x)
    print(f"  Input 1: pred={final_pred[0].tolist()}, target={y[0].tolist()}")
    print(f"  Input 2: pred={final_pred[1].tolist()}, target={y[1].tolist()}")