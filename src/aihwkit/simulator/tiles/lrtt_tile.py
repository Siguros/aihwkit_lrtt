# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT Simulator Tile with Python orchestration.

Integrates LRTTController with aihwkit tile system, providing the same interface
as rpucuda_lrtt_transfer_device.cu but implemented entirely in Python.
"""

from typing import Optional, Tuple, Any, Dict
import torch
from torch import Tensor
from torch.nn import Module

from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
# LRTTTransferCompound removed - using Python-level LRTT instead
from aihwkit.exceptions import ConfigError, TileError


class LRTTSimulatorTile(SimulatorTile, Module):
    """LR-TT simulator tile with Python orchestration.
    
    Implements the exact semantics of rpucuda_lrtt_transfer_device.cu using
    3 analog tiles (fastA, fastB, visible) orchestrated by LRTTController.
    
    Architecture:
    - tile_a (fastA): A matrix [d_size, rank] for LoRA left factor
    - tile_b (fastB): B matrix [rank, x_size] for LoRA right factor  
    - tile_c (visible): Main weights [d_size, x_size]
    
    Key features:
    - Rank-restricted LoRA-style updates with projections
    - Pulsed transfer with outer-product accumulation
    - Forward injection: y = C·x + α·A·(B·x)
    - Full scheduling and BL management support
    """
    
    def __init__(
        self, 
        d_size: int,  # out_features from AnalogLinear
        x_size: int,  # in_features from AnalogLinear 
        rpu_config: UnitCellRPUConfig, 
        dtype: RPUDataType
    ):
        """Initialize LRTT simulator tile.
        
        Args:
            d_size: Output size (out_features from AnalogLinear)
            x_size: Input size (in_features from AnalogLinear)
            rpu_config: Must contain LRTTTransferCompound device
            dtype: Data type for tiles
        """
        Module.__init__(self)
        
        self.x_size = x_size
        self.d_size = d_size
        self.dtype = dtype
        
        # Validate configuration - check for PythonLRTTDevice
        from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
        if not isinstance(getattr(rpu_config, 'device', None), PythonLRTTDevice):
            raise ConfigError("LRTTSimulatorTile requires a PythonLRTTDevice configuration")
            
        self.lrtt_config = rpu_config.device
        self.rank = self.lrtt_config.rank
        
        if self.rank <= 0 or self.rank > min(d_size, x_size):
            raise ConfigError(f"Invalid rank {self.rank} for dimensions {d_size}×{x_size}")
            
        # Extract LRTT parameters
        self.transfer_lr = getattr(self.lrtt_config, 'transfer_lr', 1.0)
        self.transfer_every = getattr(self.lrtt_config, 'transfer_every', 32)
        self.units_in_mbatch = getattr(self.lrtt_config, 'units_in_mbatch', False)
        self.lora_alpha = getattr(self.lrtt_config, 'lora_alpha', 1.0)
        self.reinit_gain = getattr(self.lrtt_config, 'reinit_gain', 0.1)
        self.correct_gradient_magnitudes = getattr(self.lrtt_config, 'correct_gradient_magnitudes', False)
        # Note: forward_inject flag is managed by the controller only
        self.rank_chunk = getattr(self.lrtt_config, 'rank_chunk', None)
        
        # Create individual tiles using unit cell devices
        unit_devices = self.lrtt_config.unit_cell_devices
        if len(unit_devices) < 3:
            # Replicate first device if not enough specified
            while len(unit_devices) < 3:
                unit_devices.append(unit_devices[0])
                
        # Tile A: fastA [d_size, rank]
        rpu_config_a = SingleRPUConfig(
            device=unit_devices[0],
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=rpu_config.update,
            tile_class=AnalogTile
        )
        self.tile_a = rpu_config_a.tile_class(d_size, self.rank, rpu_config_a)
        
        # Tile B: fastB [rank, x_size] (only rank rows needed for LoRA)
        rpu_config_b = SingleRPUConfig(
            device=unit_devices[1], 
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=rpu_config.update,
            tile_class=AnalogTile
        )
        self.tile_b = rpu_config_b.tile_class(self.rank, x_size, rpu_config_b)
        
        # Tile C: visible [d_size, x_size]  
        rpu_config_c = SingleRPUConfig(
            device=unit_devices[2],
            forward=rpu_config.forward,
            backward=rpu_config.backward, 
            update=rpu_config.update,
            tile_class=AnalogTile
        )
        self.tile_c = rpu_config_c.tile_class(d_size, x_size, rpu_config_c)
        
        # Create LRTT controller
        self.controller = LRTTController(
            tile_a=self.tile_a,
            tile_b=self.tile_b,
            tile_c=self.tile_c,
            d_size=d_size,
            x_size=x_size,
            rank=self.rank,
            transfer_lr=self.transfer_lr,
            transfer_every=self.transfer_every,
            units_in_mbatch=self.units_in_mbatch,
            lora_alpha=self.lora_alpha,
            reinit_gain=self.reinit_gain,
            correct_gradient_magnitudes=self.correct_gradient_magnitudes,
            rank_chunk=self.rank_chunk,
            forward_inject=getattr(self.lrtt_config, 'forward_inject', True)
        )
        
        # Initialize LRTT weights
        self.controller.reinit()
        
        # Hook individual tile updates to route through controller
        self._hook_tile_updates()
        
    def _hook_tile_updates(self) -> None:
        """Hook individual tile update methods to route through controller.
        
        When AnalogSGD calls update on individual tiles, we intercept
        and route through the controller for proper LRTT updates.
        """
        # Store original update methods
        self.tile_a._orig_update = self.tile_a.update
        self.tile_b._orig_update = self.tile_b.update
        self.tile_c._orig_update = self.tile_c.update
        
        # Track if we've already handled this batch
        self._update_handled = False
        
        def hooked_update(tile_name):
            def update_wrapper(x_input, d_input, *args, **kwargs):
                # Prevent double updates - only handle once per batch
                if self._update_handled:
                    return None
                    
                # Tile C gets the full inputs, use those for LRTT update
                if tile_name == 'tile_c':
                    self._update_handled = True  # Mark as handled
                    
                    # Get learning rate
                    lr = self.tile_c.get_learning_rate()
                    
                    # Route through controller for proper LRTT update
                    self.controller.ab_weight_update(
                        x=x_input,  # This is the full [batch, x_size] input
                        d=d_input,  # This is the full [batch, d_size] gradient
                        lr=lr,
                        in_trans=False,
                        out_trans=False
                    )
                    
                    # Check for transfer
                    if self.controller.should_transfer():
                        self.controller.ab_weight_transfer()
                    
                # Don't call original update on any tile - LRTT handles all updates
                return None
            return update_wrapper
        
        # Replace update methods
        self.tile_a.update = hooked_update('tile_a')
        self.tile_b.update = hooked_update('tile_b')
        self.tile_c.update = hooked_update('tile_c')
        
    def _reset_update_flag(self) -> None:
        """Reset the update handled flag for next batch."""
        self._update_handled = False
        
    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Forward pass with LRTT forward injection.
        
        Args:
            x_input: Input tensor 
            bias: Bias flag (not supported)
            in_trans: Input transposed
            out_trans: Output transposed
            is_test: Test mode (affects forward injection)
            non_blocking: Non-blocking flag
            
        Returns:
            Output tensor
        """
        # Reset update flag for this forward pass
        self._reset_update_flag()
        
        if bias:
            raise TileError("LRTT does not support bias")
            
        # Single source of truth: Use controller's forward_inject_enabled flag only
        # This avoids confusion from multiple forward_inject flags
        if self.controller.forward_inject_enabled:
            return self.controller.forward_inject(x_input, out_trans=out_trans, in_trans=in_trans)
        else:
            # Fallback to visible-only forward when disabled
            # Handle transpose manually since AnalogTile doesn't support transpose flags
            x = x_input.t() if in_trans else x_input
            y = self.tile_c.forward(x)
            return y.t() if out_trans else y
    
    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """LRTT backward pass using only analog tile operations.
        
        Computes: x_grad = C^T @ d + α * B^T @ (A^T @ d)
        All operations use tile.backward() to ensure proper analog constraints.
        """
        if bias:
            raise TileError("LRTT does not support bias")
            
        # 1) Input to batch-first
        d_bf = d_input.t() if in_trans else d_input  # [batch, d_size]
        
        # 2) C^T·d, A^T·d, B^T·(A^T·d) — all using tile backward
        xg_c = self.tile_c.backward(d_bf)   # [batch, x_size]
        da = self.tile_a.backward(d_bf)     # [batch, rank]
        xg_ab = self.tile_b.backward(da)    # [batch, x_size]
        
        # 3) Composition
        x_grad = xg_c + self.lora_alpha * xg_ab
        
        # 4) Output transpose
        return x_grad.t() if out_trans else x_grad
    
    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """LRTT update: A/B LoRA updates + periodic transfer.
        
        Args:
            x_input: Input tensor
            d_input: Error tensor
            bias: Bias flag (not supported)
            in_trans: Input transposed
            out_trans: Output transposed  
            non_blocking: Non-blocking flag
        """
        if bias:
            raise TileError("LRTT does not support bias")
            
        # Prevent double updates
        if self._update_handled:
            return None
        self._update_handled = True
        
        # Get current learning rate (assuming all tiles have same LR)
        lr = self.get_learning_rate()
        
        # Perform A/B LoRA-style updates with projections
        self.controller.ab_weight_update(
            x=x_input,
            d=d_input, 
            lr=lr,
            in_trans=in_trans,
            out_trans=out_trans
        )
        
        # Check for transfer
        if self.controller.should_transfer():
            self.controller.ab_weight_transfer()
            
    def get_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get visible weights (source of truth), matching CUDA semantics.
        
        Returns:
            Tuple of (visible_weights, None)
        """
        # CRITICAL: Return visible weights only, not effective weights
        # This matches CUDA where visible (C) is the source of truth
        return self.tile_c.get_weights()
    
    def get_effective_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get effective LRTT weights: W_eff = W_visible + α * A @ B.
        
        This is a separate method for when effective weights are explicitly needed.
        
        Returns:
            Tuple of (effective_weights, None)
        """
        from aihwkit.linalg.lrtt_kernels import compose_lrtt_weights
        
        # Get individual component weights
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_weights = self.tile_b.get_weights()[0]        # [rank, x_size]
        
        # Compose effective weights
        W_eff = compose_lrtt_weights(
            visible_weights, A_weights, B_weights, 
            self.lora_alpha, self.rank
        )
        
        return W_eff, None
        
    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Set visible weights (source of truth), A/B remain unchanged.
        
        This matches CUDA where visible weights are the primary storage.
        
        Args:
            weight: Weight tensor [d_size, x_size]
            bias: Bias tensor (not supported)
        """
        if bias is not None:
            raise TileError("LRTT does not support bias")
            
        # Set visible weights only, preserve A/B state
        self.tile_c.set_weights(weight, None)
        
    def get_lrtt_component_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get individual LRTT component weights.
        
        Returns:
            Tuple of (visible_weights, A_weights, B_lr_weights)
        """
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]             # [rank, x_size]
        
        return visible_weights, A_weights, B_lr
        
    def set_lrtt_component_weights(
        self, 
        visible: Tensor, 
        A: Tensor, 
        B_lr: Tensor
    ) -> None:
        """Set individual LRTT component weights.
        
        Args:
            visible: Visible weights [d_size, x_size]
            A: A weights [d_size, rank]
            B_lr: B weights [rank, x_size] (will be placed in first rank rows)
        """
        # Set visible weights
        self.tile_c.set_weights(visible, None)
        
        # Set A weights
        self.tile_a.set_weights(A, None)
        
        # Set B weights (B tile is already [rank, x_size], no expansion needed)
        self.tile_b.set_weights(B_lr, None)
        
    def get_x_size(self) -> int:
        """Get input size."""
        return self.x_size
        
    def get_d_size(self) -> int:
        """Get output size."""
        return self.d_size
        
    def get_learning_rate(self) -> float:
        """Get learning rate from visible tile."""
        return self.tile_c.get_learning_rate()
        
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate for all tiles."""
        self.tile_a.set_learning_rate(learning_rate)
        self.tile_b.set_learning_rate(learning_rate)
        self.tile_c.set_learning_rate(learning_rate)
        
    def get_hidden_parameters(self) -> Tensor:
        """Get concatenated hidden parameters from all tiles."""
        params_a = self.tile_a.get_hidden_parameters()
        params_b = self.tile_b.get_hidden_parameters() 
        params_c = self.tile_c.get_hidden_parameters()
        
        return torch.cat([params_a, params_b, params_c])
        
    def set_hidden_parameters(self, data: Tensor) -> None:
        """Set hidden parameters for all tiles."""
        # Split data based on tile parameter counts
        params_a = self.tile_a.get_hidden_parameters()
        params_b = self.tile_b.get_hidden_parameters()
        params_c = self.tile_c.get_hidden_parameters()
        
        size_a = params_a.numel()
        size_b = params_b.numel()
        size_c = params_c.numel()
        
        if data.numel() != size_a + size_b + size_c:
            raise TileError(f"Hidden parameter size mismatch: expected {size_a + size_b + size_c}, got {data.numel()}")
            
        self.tile_a.set_hidden_parameters(data[:size_a])
        self.tile_b.set_hidden_parameters(data[size_a:size_a + size_b])
        self.tile_c.set_hidden_parameters(data[size_a + size_b:])
        
    def decay_weights(self, alpha: float = 0.0) -> None:
        """Apply weight decay to all tiles."""
        self.tile_a.decay_weights(alpha)
        self.tile_b.decay_weights(alpha) 
        self.tile_c.decay_weights(alpha)
        
    def diffuse_weights(self, alpha: float = 0.0) -> None:
        """Apply weight diffusion to all tiles."""
        self.tile_a.diffuse_weights(alpha)
        self.tile_b.diffuse_weights(alpha)
        self.tile_c.diffuse_weights(alpha)
        
    def clip_weights(self, clip_type: str = "", sigma: float = 0.0) -> None:
        """Apply weight clipping to all tiles.""" 
        self.tile_a.clip_weights(clip_type, sigma)
        self.tile_b.clip_weights(clip_type, sigma)
        self.tile_c.clip_weights(clip_type, sigma)
        
    def reset_columns(self, start_column_idx: int = 0, num_columns: int = 1, sigma: float = 1.0) -> None:
        """Reset columns in visible tile."""
        # Only reset visible tile columns (A/B managed by controller)
        self.tile_c.reset_columns(start_column_idx, num_columns, sigma)
        
    def get_brief_info(self) -> str:
        """Get brief tile information."""
        return f"LRTTSimulatorTile({self.d_size}, {self.x_size}, rank={self.rank})"
        
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"d_size={self.d_size}, x_size={self.x_size}, rank={self.rank}, " \
               f"transfer_every={self.transfer_every}, lora_alpha={self.lora_alpha}"
               
    def get_controller_state(self) -> Dict[str, Any]:
        """Get LRTT controller state for debugging/monitoring."""
        return self.controller.get_state_dict()
        
    def manual_transfer(self) -> None:
        """Manually trigger A⊗B -> visible transfer (for testing)."""
        self.controller.ab_weight_transfer()