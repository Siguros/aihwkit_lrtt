#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Simple example demonstrating Python-level LRTT functionality.

This example shows the basic usage of LRTT (Low-Rank Transfer Tiki-Taka)
for a single layer with detailed inspection of the internal states.
"""

import torch
import torch.nn.functional as F
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    lrtt_idealized_config,
    lrtt_constant_step_config,
    PythonLRTTDevice,
    PythonLRTTRPUConfig
)


def inspect_lrtt_layer(layer, name="Layer"):
    """Inspect LRTT layer internals."""
    print(f"\n=== {name} Inspection ===")
    
    # Check if it's an LRTT tile
    tile = layer.analog_module
    if not hasattr(tile, 'controller'):
        print("Not an LRTT tile")
        return
    
    controller = tile.controller
    
    # Get component weights
    visible_weights, A_weights, B_lr = tile.get_lrtt_component_weights()
    
    print(f"Visible weights shape: {visible_weights.shape}")
    print(f"A weights shape: {A_weights.shape}")
    print(f"B_lr weights shape: {B_lr.shape}")
    print(f"Visible norm: {visible_weights.norm().item():.4f}")
    print(f"A norm: {A_weights.norm().item():.4f}")
    print(f"B norm: {B_lr.norm().item():.4f}")
    
    # Controller stats
    print(f"\nController stats:")
    print(f"  Rank: {controller.rank}")
    print(f"  Transfer every: {controller.transfer_every}")
    print(f"  Transfer counter: {controller.transfer_counter}")
    print(f"  Transfers done: {controller.num_transfers}")
    print(f"  A updates: {controller.num_a_updates}")
    print(f"  B updates: {controller.num_b_updates}")
    
    # Effective weights
    effective_weights = tile.get_effective_weights()[0]
    print(f"\nEffective weights norm: {effective_weights.norm().item():.4f}")
    print(f"Difference from visible: {(effective_weights - visible_weights).norm().item():.4f}")


def main():
    """Main example function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Create different LRTT configurations
    print("\n" + "="*60)
    print("1. TESTING DIFFERENT LRTT CONFIGURATIONS")
    print("="*60)
    
    # Idealized config
    config_ideal = lrtt_idealized_config(rank=4, transfer_every=10, lora_alpha=2.0)
    print(f"\nIdealized config: {config_ideal.get_brief_info()}")
    
    # Constant step config
    config_constant = lrtt_constant_step_config(rank=4, transfer_every=10, dw_min=0.01)
    print(f"Constant step config: {config_constant.get_brief_info()}")
    
    # Custom config
    custom_device = PythonLRTTDevice(
        rank=2,
        transfer_every=5,
        transfer_lr=0.5,
        lora_alpha=1.5,
        reinit_gain=0.05,
        forward_inject=True,
        correct_gradient_magnitudes=True
    )
    config_custom = PythonLRTTRPUConfig(device=custom_device)
    print(f"Custom config: {config_custom.get_brief_info()}")
    
    # 2. Create a simple layer with LRTT
    print("\n" + "="*60)
    print("2. CREATING LRTT LAYER")
    print("="*60)
    
    layer = AnalogLinear(10, 5, bias=False, rpu_config=config_constant)
    layer = layer.to(device)
    
    inspect_lrtt_layer(layer, "Initial state")
    
    # 3. Perform training steps
    print("\n" + "="*60)
    print("3. TRAINING WITH LRTT")
    print("="*60)
    
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Generate random data
    x = torch.randn(4, 10, device=device)  # batch_size=4, input_size=10
    target = torch.randn(4, 5, device=device)  # batch_size=4, output_size=5
    
    print("\nTraining for 15 steps (transfer_every=10)...")
    losses = []
    
    for step in range(15):
        # Forward pass
        output = layer(x)
        loss = F.mse_loss(output, target)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check for transfer
        controller = layer.analog_module.controller
        if step > 0 and controller.num_transfers > (step - 1) // 10:
            print(f"  Step {step+1}: Loss = {loss.item():.4f} [TRANSFER OCCURRED]")
        else:
            print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    inspect_lrtt_layer(layer, "After training")
    
    # 4. Test forward injection
    print("\n" + "="*60)
    print("4. TESTING FORWARD INJECTION")
    print("="*60)
    
    # Get outputs with and without forward injection
    layer.eval()
    with torch.no_grad():
        # With forward injection (default)
        output_with_inject = layer(x)
        
        # Get visible-only output (simulate forward_inject=False)
        visible_weights = layer.analog_module.get_weights()[0].to(device)
        output_visible_only = F.linear(x, visible_weights, None)
        
        # Get effective weights output
        effective_weights = layer.analog_module.get_effective_weights()[0].to(device)
        output_effective = F.linear(x, effective_weights, None)
    
    print(f"Output with forward injection norm: {output_with_inject.norm().item():.4f}")
    print(f"Output with visible weights only norm: {output_visible_only.norm().item():.4f}")
    print(f"Output with effective weights norm: {output_effective.norm().item():.4f}")
    print(f"Difference (inject vs effective): {(output_with_inject - output_effective).norm().item():.6f}")
    
    # 5. Demonstrate rank decomposition
    print("\n" + "="*60)
    print("5. RANK DECOMPOSITION ANALYSIS")
    print("="*60)
    
    visible_weights, A_weights, B_lr = layer.analog_module.get_lrtt_component_weights()
    
    # Compute A @ B manually
    AB_product = torch.matmul(A_weights[:, :config_constant.device.rank], B_lr)
    
    # Compute effective weights manually
    manual_effective = visible_weights + config_constant.device.lora_alpha * AB_product
    
    # Compare with tile's effective weights
    tile_effective = layer.analog_module.get_effective_weights()[0]
    
    print(f"Rank of decomposition: {config_constant.device.rank}")
    print(f"A @ B product norm: {AB_product.norm().item():.4f}")
    print(f"Manual effective weights norm: {manual_effective.norm().item():.4f}")
    print(f"Tile effective weights norm: {tile_effective.norm().item():.4f}")
    print(f"Difference: {(manual_effective - tile_effective).norm().item():.6f}")
    
    print("\n" + "="*60)
    print("✅ LRTT EXAMPLE COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()