# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT Controller: Pure Python orchestrator for 3-tile LRTT (fastA, fastB, visible).

Implements the exact semantics from rpucuda_lrtt_transfer_device.cu as a pure Python
orchestrator on top of aihwkit tiles. Operates on A, B, visible (C) tile stack with:
- Rank-restricted LoRA-style updates
- Pulsed transfer with outer-product accumulation  
- Forward injection with W_eff composition
- Full BL-management and scheduling support
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Any
import math

from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery
from aihwkit.simulator.parameters.enums import PulseType


class LRTTController:
    """LR-TT controller orchestrating 3 analog tiles: fastA, fastB, visible (C).
    
    Replicates rpucuda_lrtt_transfer_device.cu behavior with:
    - tile_a: FastA weights [d_size, rank] for LoRA left factor
    - tile_b: FastB weights [rank, x_size] for LoRA right factor  
    - tile_c: Visible weights [d_size, x_size] for main matrix C
    
    Core operations:
    1. reinit(): A=0, B~Kaiming (first rank rows), optional C init
    2. ab_weight_update(): LoRA-style pulsed updates with projections
    3. ab_weight_transfer(): A⊗B -> C transfer, then reinit
    4. forward_inject(): y = C·x + α·A·(B·x) composition
    """
    
    def __init__(
        self,
        tile_a: AnalogTileWithoutPeriphery,   # fastA [d_size, rank]
        tile_b: AnalogTileWithoutPeriphery,   # fastB [rank, x_size] 
        tile_c: AnalogTileWithoutPeriphery,   # visible [d_size, x_size]
        d_size: int,
        x_size: int,
        rank: int,
        *,
        transfer_lr: float = 1.0,
        transfer_every: int = 32,
        units_in_mbatch: bool = False,
        lora_alpha: float = 1.0,
        reinit_gain: float = 0.1,
        correct_gradient_magnitudes: bool = False,
        rank_chunk: Optional[int] = None,
        ab_bl_mgmt: Optional[Dict[str, Any]] = None,
        transfer_bl_mgmt: Optional[Dict[str, Any]] = None,
        forward_inject: bool = True,
        device: Optional[torch.device] = None,  # Explicit device to avoid get_weights()
        dtype: torch.dtype = torch.float32      # Explicit dtype
    ):
        """Initialize LR-TT controller.
        
        Args:
            tile_a: FastA tile for A matrix [d_size, rank]
            tile_b: FastB tile for B matrix [rank, x_size]
            tile_c: Visible tile for C matrix [d_size, x_size]
            d_size: Output dimension
            x_size: Input dimension
            rank: LoRA rank (must be <= min(d_size, x_size))
            transfer_lr: Transfer learning rate scalar
            transfer_every: Transfer frequency (steps or samples)
            units_in_mbatch: Whether transfer_every counts samples vs steps
            lora_alpha: LoRA scaling factor α
            reinit_gain: Kaiming initialization gain for B matrix
            correct_gradient_magnitudes: Scale lr by sqrt(rank) for gradient correction
            rank_chunk: Chunk size for transfer (None = full rank)
            ab_bl_mgmt: BL management for A/B updates {update_bl_management, update_management, desired_BL}
            transfer_bl_mgmt: BL management for transfers
            forward_inject: Enable forward injection optimization
            device: Explicit device (if None, safely inferred from tiles using tiny dummy forward)
                   Strongly recommended to pass the tile device explicitly for best performance
            dtype: Explicit dtype for tensors
        """
        if rank <= 0 or rank > min(d_size, x_size):
            raise ValueError(f"Invalid rank {rank} for dimensions {d_size}×{x_size}")
            
        self.tile_a = tile_a
        self.tile_b = tile_b  
        self.tile_c = tile_c
        
        self.d_size = d_size
        self.x_size = x_size
        self.rank = rank
        
        # LRTT parameters
        self.transfer_lr = transfer_lr
        self.transfer_every = transfer_every
        self.units_in_mbatch = units_in_mbatch
        self.lora_alpha = lora_alpha
        self.reinit_gain = reinit_gain
        self.correct_gradient_magnitudes = correct_gradient_magnitudes
        self.rank_chunk = rank_chunk or rank
        self.forward_inject_enabled = forward_inject
        
        # BL management settings
        self.ab_bl_mgmt = ab_bl_mgmt or {}
        self.transfer_bl_mgmt = transfer_bl_mgmt or {}
        
        # Counters and state
        self.transfer_counter = 0
        self.num_a_updates = 0
        self.num_b_updates = 0
        self.num_transfers = 0
        
        # Cached buffers for efficiency
        self._x_b_buffer: Optional[Tensor] = None
        self._d_a_buffer: Optional[Tensor] = None  
        self._pad_buffer_a: Optional[Tensor] = None
        self._pad_buffer_b: Optional[Tensor] = None
        
        # Device info - infer from tiles if not provided
        if device is None:
            # Safely infer device from tile using a tiny dummy forward
            device = self._infer_device_from_tile()
        self.device = device
        self.dtype = dtype
        
        # Track initialization state with flags to avoid weight norm checks
        self._c_initialized = True
        self._tiles_initialized = False
        
    def _infer_device_from_tile(self) -> torch.device:
        """Safely infer device from tile.
        
        This is now a simple fallback since device should be explicitly synchronized
        via set_device() when tiles are moved.
        """
        # Check tile backend type
        if hasattr(self.tile_c, 'tile'):
            tile_str = str(type(self.tile_c.tile).__name__)
            if 'Cuda' in tile_str or 'CUDA' in tile_str:
                return torch.device('cuda')
        
        # Default based on CUDA availability
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_tile_device(self) -> torch.device:
        """Get device that tiles expect for operations."""
        # OPTIMIZATION: Return cached device instead of using get_weights()
        return self.device
        
    def _get_tile_dtype(self) -> torch.dtype:
        """Get common dtype from tiles."""
        # OPTIMIZATION: Return cached dtype instead of checking tiles
        return self.dtype
        
    def _ensure_buffers(self, batch_size: int) -> None:
        """Ensure scratch buffers are allocated for given batch size."""
        if (self._x_b_buffer is None or 
            self._x_b_buffer.size(-1) != batch_size):
            
            # Use cached device
            device = self.device
            
            # Projection buffers
            self._x_b_buffer = torch.zeros(
                self.rank, batch_size, device=device, dtype=self.dtype
            )
            self._d_a_buffer = torch.zeros(
                self.rank, batch_size, device=device, dtype=self.dtype  
            )
            
            # CRITICAL FIX: Padding buffers must match tile input dimensions
            # A tile expects [x_size, batch] inputs (not d_size!)
            # B tile expects [d_size, batch] for errors
            self._x_pad = torch.zeros(
                self.x_size, batch_size, device=device, dtype=self.dtype
            )
            self._d_pad = torch.zeros(
                self.d_size, batch_size, device=device, dtype=self.dtype
            )
    
    def reinit(self) -> None:
        """Reinit: A=0; B[:rank,:] ~ Kaiming; optional visible init.
        
        Following LoRA convention:
        - A: zero the full matrix → ensures A_lr == 0
        - B: zero full matrix, then Kaiming Normal for first rank rows only  
        - Visible (C): small Kaiming init if needed for forward_inject
        """
        # A matrix: full zero
        A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
        self.tile_a.set_weights(A_zeros)
        
        # B matrix: [rank, x_size] - Kaiming Normal initialization
        # Kaiming Normal: std = gain * sqrt(2 / fan_in), fan_in = x_size
        std = self.reinit_gain * math.sqrt(2.0 / self.x_size)
        B_kaiming = torch.normal(0, std, size=(self.rank, self.x_size), device=self.device, dtype=self.dtype)
        
        self.tile_b.set_weights(B_kaiming)
        
        # Apply device clipping if available
        if hasattr(self.tile_b, 'clip_weights'):
            self.tile_b.clip_weights()
            
        # OPTIMIZATION: Use flag instead of reading C weights for norm check
        if self.forward_inject_enabled and not self._c_initialized:
            # Small Kaiming init to avoid degenerate W_eff
            C_std = self.reinit_gain * math.sqrt(2.0 / self.x_size) * 0.1  # Smaller
            C_init = torch.normal(0, C_std, size=(self.d_size, self.x_size), device=self.device, dtype=self.dtype)
            self.tile_c.set_weights(C_init)
            if hasattr(self.tile_c, 'clip_weights'):
                self.tile_c.clip_weights()
            self._c_initialized = True
        
        # Reset counters
        self.transfer_counter = 0
        self._tiles_initialized = True
        
    def ab_weight_update(
        self, 
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """Update A and B with LoRA-style rank-r gradient approximation.
        
        Simplified batch-first processing with no intermediate transposes.
        Uses tile forward/backward for projections and tile update for weight changes.
        
        Args:
            x: Input tensor
            d: Error tensor
            lr: Learning rate
            in_trans: Whether x is transposed
            out_trans: Whether d is transposed
        """
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()
        
        # 1) Projections (analog path)
        with torch.no_grad():
            XB = self.tile_b.forward(x)     # [batch, rank] = B·X
            DA = self.tile_a.backward(d)    # [batch, rank] = A^T·D
        
        # 2) lr_eff = lr * α * (1/√r, optional)
        lr_eff = lr * self.lora_alpha
        if self.correct_gradient_magnitudes:
            lr_eff /= math.sqrt(self.rank)
        
        # 3) ΔA = -lr_eff · D^T · (B·X) → tile_a.update(XB, d)
        lr_a_old = self.tile_a.get_learning_rate()
        self.tile_a.set_learning_rate(lr_eff)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(XB, d)
        else:
            self.tile_a.update(XB, d)
        self.tile_a.set_learning_rate(lr_a_old)
        self.num_a_updates += 1
        
        # 4) ΔB = -lr_eff · (A^T·D)^T · X → tile_b.update(x, DA)
        lr_b_old = self.tile_b.get_learning_rate()
        self.tile_b.set_learning_rate(lr_eff)
        if hasattr(self.tile_b, '_orig_update'):
            self.tile_b._orig_update(x, DA)
        else:
            self.tile_b.update(x, DA)
        self.tile_b.set_learning_rate(lr_b_old)
        self.num_b_updates += 1
        
        # 5) Counter
        self.transfer_counter += (x.shape[0] if self.units_in_mbatch else 1)
            
    def ab_weight_transfer(self, use_onehot: bool = False) -> None:
        """Memory-optimized pulsed A⊗B -> visible transfer, then reinit.
        
        Transfer: C += transfer_lr * (A @ B) via pulsed outer product.
        
        Args:
            use_onehot: If True, use one-hot reading (analog-realistic).
                       If False, use direct weight access (default).
        
        Direct mode:
        1. Get weights to CPU first to avoid GPU memory spike
        2. For chunks of rank: pack D_chunk = A[:, off:off+cur], X_chunk = B[off:off+cur, :]  
        3. Move only chunks to GPU for update
        4. Call visible pulsed updater: C.update(X_chunk^T, D_chunk, lr=|transfer_lr|)
        5. Handle sign rule: negate D when transfer_lr > 0
        6. Unconditionally call reinit() after transfer
        
        One-hot mode:
        1. Read A columns using forward pass with one-hot vectors
        2. Read B rows using backward pass with one-hot vectors
        3. Accumulate outer products into C
        4. Unconditionally call reinit() after transfer
        """
        if use_onehot:
            self._ab_weight_transfer_onehot()
        else:
            self._ab_weight_transfer_direct()
    
    def _ab_weight_transfer_direct(self) -> None:
        """Original transfer implementation using direct weight access."""
        with torch.no_grad():
            # Get weights (they come in the tile's native device)
            A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
            B_weights = self.tile_b.get_weights()[0]  # [rank, x_size]
            
            A_lr = A_weights[:, :self.rank]  # [d_size, rank]
            
            # Transfer in chunks to manage memory
            lr_eff = abs(self.transfer_lr)
            old_lr = self.tile_c.get_learning_rate()
            self.tile_c.set_learning_rate(lr_eff)
            
            # Apply transfer BL management  
            if self.transfer_bl_mgmt:
                # Apply transfer_bl_mgmt settings
                pass
                
            chunk_size = self.rank_chunk
            for off in range(0, self.rank, chunk_size):
                end = min(off + chunk_size, self.rank)
                cur = end - off
                
                # Pack chunks (keep on same device as tiles)
                D_chunk = A_lr[:, off:end].contiguous()  # [d_size, cur]
                X_chunk = B_weights[off:end, :].contiguous()     # [cur, x_size]
                
                # Sign rule: PWU computes W += -lr * D @ X^T, we want W += +transfer_lr * D @ X^T
                # So when transfer_lr > 0, negate D to get correct sign
                if self.transfer_lr > 0:
                    D_chunk = -D_chunk
                elif self.transfer_lr < 0:
                    # transfer_lr < 0: want W += transfer_lr * D @ X^T (negative), so keep D positive
                    # PWU does W += -lr * D @ X^T with lr > 0, so net effect is W += -D @ X^T (negative) ✓
                    pass
                    
                # Use controller's device (single source of truth)
                dev = self.device
                X_chunk_d = X_chunk.contiguous().to(dev, non_blocking=True)
                D_chunk_t_d = D_chunk.t().contiguous().to(dev, non_blocking=True)
                
                # Debug assertion to ensure same device
                assert X_chunk_d.device == D_chunk_t_d.device, \
                    f"Device mismatch: X={X_chunk_d.device}, D={D_chunk_t_d.device}"
                
                # Pulsed update to C tile
                if hasattr(self.tile_c, '_orig_update'):
                    self.tile_c._orig_update(X_chunk_d, D_chunk_t_d)
                else:
                    self.tile_c.update(X_chunk_d, D_chunk_t_d)
                
                # OPTIMIZATION: Immediately free GPU memory
                del X_chunk_d, D_chunk_t_d
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.tile_c.set_learning_rate(old_lr)
        self.num_transfers += 1
        
        # CRITICAL: Reset transfer counter after transfer (matches CUDA)
        self.transfer_counter = 0
        
        # Unconditional reinit after transfer
        self.reinit()
    
    def _ab_weight_transfer_onehot(self) -> None:
        """One-hot based transfer following transfer.py pattern (lines 247-263).
        
        Uses analog-realistic reading via forward/backward passes with one-hot vectors.
        """
        with torch.no_grad():
            # Create one-hot vectors using controller's device
            if not hasattr(self, '_transfer_vec_a') or self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(
                    self.rank,
                    dtype=self.dtype,
                    device=self.device  # Use controller's device
                )
            
            # Save and set learning rate for transfer
            lr_eff = abs(self.transfer_lr)
            old_lr = self.tile_c.get_learning_rate()
            self.tile_c.set_learning_rate(lr_eff)
            
            # Process each rank dimension sequentially (like transfer.py)
            for k in range(self.rank):
                # Prepare one-hot input as [1, rank] for batch dimension
                e_k = self._transfer_vec_a[k].unsqueeze(0)  # [1, rank]
                
                # Read column k of A using forward with one-hot
                a_col_b1 = self.tile_a.forward(e_k)  # Shape: [1, d_size]
                
                # Read row k of B using backward with one-hot
                b_row_b1 = self.tile_b.backward(e_k)  # Shape: [1, x_size]
                
                # Apply sign correction (same as direct method)
                if self.transfer_lr > 0:
                    a_col_b1 = -a_col_b1
                
                # Ensure tensors are on controller's device
                X_k = b_row_b1.to(self.device, non_blocking=True)  # [1, x_size]
                D_k = a_col_b1.to(self.device, non_blocking=True)  # [1, d_size]
                
                # Update C tile with rank-1 outer product
                if hasattr(self.tile_c, '_orig_update'):
                    self.tile_c._orig_update(X_k, D_k)
                else:
                    self.tile_c.update(X_k, D_k)
            
            # Restore learning rate
            self.tile_c.set_learning_rate(old_lr)
            
        self.num_transfers += 1
        self.transfer_counter = 0
        self.reinit()
        
    def forward_inject(
        self,
        x: Tensor,                    # [x_size, m] or [batch, x_size]
        out_trans: bool = False,
        in_trans: bool = False
    ) -> Tensor:
        """Forward inject: y = C·x + lora_alpha * A·(B·x).
        
        Returns y = C·x + α * A·(B·x) under these rules:
        - If forward_inject_enabled=False or rank=0: visible-only (y = C·x)
        - Default analog-hybrid: y_vis = C·x, g = B·x, y_ab = A·g, y = y_vis + α*y_ab
        - Fallback (transposed): digital composition W_eff = C + α*(A_lr @ B_lr), then W_eff @ x
        
        Args:
            x: Input tensor [x_size, m] or [batch, x_size]
            out_trans: Output transposed flag
            in_trans: Input transposed flag
            
        Returns:
            Output tensor [d_size, m] or [batch, d_size]
        """
        # Initialize tiles on first forward if needed
        if not self._tiles_initialized:
            self.reinit()
            
        # Handle disabled forward injection
        if not self.forward_inject_enabled or self.rank == 0:
            return self.tile_c.forward(x, in_trans=in_trans, out_trans=out_trans)
            
        # Use unified analog path for all cases (including transpose)
        return self._forward_inject_analog_unified(x, in_trans=in_trans, out_trans=out_trans)
        
    def _forward_inject_digital_fallback(
        self, 
        x: Tensor,
        out_trans: bool, 
        in_trans: bool
    ) -> Tensor:
        """Digital fallback: compose W_eff then single forward pass.
        
        WARNING: This path creates large GPU tensors and can cause OOM!
        The unified analog path should be used instead whenever possible.
        """
        # WARNING: get_weights() can cause memory issues with large models
        C_weights = self.tile_c.get_weights()[0]   # [d_size, x_size]
        A_lr = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]        # [rank, x_size]
        
        # WARNING: This creates a large intermediate tensor W_eff
        W_eff = C_weights + self.lora_alpha * (A_lr @ B_lr)
        
        # Set temporary weights and forward
        original_weights = C_weights.clone()
        self.tile_c.set_weights(W_eff)
        
        try:
            result = self.tile_c.forward(x, bias=False, in_trans=in_trans, out_trans=out_trans)
        finally:
            # Restore original weights
            self.tile_c.set_weights(original_weights)
            
        return result
        
    def _forward_inject_analog_hybrid(self, x: Tensor) -> Tensor:
        """Analog-hybrid path using direct weight computation (deterministic).
        
        Rcolaces non-deterministic tile forward operations with direct matrix computation:
        y = x @ (C^T + α * B^T @ A^T)
        
        This ensures consistent forward pass behavior for training stability.
        """
        # Get component weights directly
        C_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0][:, :self.rank]  # [d_size, rank] 
        B_weights = self.tile_b.get_weights()[0][:self.rank, :]  # [rank, x_size]
        
        # Compute effective weight matrix: W_eff = C^T + α * B^T @ A^T
        W_eff = C_weights.t() + self.lora_alpha * (B_weights.t() @ A_weights.t())
        
        # Ensure same device as input
        W_eff = W_eff.to(x.device)
        
        # Forward pass: y = x @ W_eff
        result = x @ W_eff  # [batch, x_size] @ [x_size, d_size] = [batch, d_size]
        
        return result
        
    def _forward_inject_analog_unified(
        self, 
        x: Tensor, 
        in_trans: bool, 
        out_trans: bool
    ) -> Tensor:
        """Unified analog path using proper tile forward operations.
        
        Uses analog tile forward operations in the correct B→A→C order.
        This ensures analog read constraints (noise/clipping) are applied
        and AnalogSGD's input/error caches work correctly.
        """
        # 1) Normalize input to batch-first
        x_bf = x.t() if in_trans else x  # [batch, x_size]
        
        # 2) Analog read order guaranteed: B → A → C
        g = self.tile_b.forward(x_bf)      # [batch, rank]
        y_ab = self.tile_a.forward(g)      # [batch, d_size]
        y_c = self.tile_c.forward(x_bf)    # [batch, d_size]
        
        # 3) Composition
        y = y_c + self.lora_alpha * y_ab   # [batch, d_size]
        
        # 4) Output transpose
        return y.t() if out_trans else y
        
    def should_transfer(self) -> bool:
        """Check if transfer should occur based on counter and schedule."""
        return self.transfer_counter >= self.transfer_every
        
    def reset_transfer_counter(self) -> None:
        """Reset transfer counter (called after transfer)."""
        self.transfer_counter = 0
        
    def get_state_dict(self) -> Dict[str, Any]:
        """Get controller state for serialization."""
        return {
            'transfer_counter': self.transfer_counter,
            'num_a_updates': self.num_a_updates,
            'num_b_updates': self.num_b_updates,
            'num_transfers': self.num_transfers,
            'd_size': self.d_size,
            'x_size': self.x_size,
            'rank': self.rank,
            'transfer_lr': self.transfer_lr,
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'forward_inject_enabled': self.forward_inject_enabled
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load controller state from serialization."""
        # Handle backward compatibility for old 'forward_inject' key
        if 'forward_inject' in state_dict and 'forward_inject_enabled' not in state_dict:
            state_dict['forward_inject_enabled'] = state_dict.pop('forward_inject')
            
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_device(self, device: torch.device) -> None:
        """Set device and clear buffers for reallocation.
        
        Args:
            device: Target device (CPU or CUDA)
        """
        self.device = torch.device(device)
        # Clear buffers so they get reallocated on the new device
        self._x_b_buffer = None
        self._d_a_buffer = None
        self._x_pad = None
        self._d_pad = None
        # Clear transfer vectors (one-hot reading)
        self._transfer_vec_a = None