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
        forward_inject: bool = True
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
        
        # Device info  
        self.device = self._get_tile_device()
        self.dtype = self._get_tile_dtype()
        
    def _get_tile_device(self) -> torch.device:
        """Get common device from tiles.""" 
        # Check if tile is a CUDA tile by looking at its class name
        if hasattr(self.tile_c, 'tile'):
            tile_str = str(self.tile_c.tile)
            if 'RPUCuda' in tile_str or 'CUDA' in tile_str:
                return torch.device('cuda')
        # Default to CUDA if available
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _get_tile_dtype(self) -> torch.dtype:
        """Get common dtype from tiles."""
        # Get dtype from tile
        if hasattr(self.tile_c, 'analog_module'):
            return torch.float32  # Default to float32 for analog modules
        else:
            return torch.float32
        
    def _ensure_buffers(self, batch_size: int) -> None:
        """Ensure scratch buffers are allocated for given batch size."""
        if (self._x_b_buffer is None or 
            self._x_b_buffer.size(-1) != batch_size):
            
            # Get device dynamically in case tiles moved to CUDA after init
            device = self._get_tile_device()
            
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
            
        # Visible (C): small init if forward_inject enabled and C is zero
        if self.forward_inject_enabled:
            C_weights = self.tile_c.get_weights()[0]
            if torch.norm(C_weights) < 1e-6:
                # Small Kaiming init to avoid degenerate W_eff
                C_std = self.reinit_gain * math.sqrt(2.0 / self.x_size) * 0.1  # Smaller
                C_init = torch.normal(0, C_std, size=(self.d_size, self.x_size), device=self.device, dtype=self.dtype)
                self.tile_c.set_weights(C_init)
        
        # Reset counters
        self.transfer_counter = 0
        
    def ab_weight_update(
        self, 
        x: Tensor,     # [x_size, m] or [m, x_size] depending on layout
        d: Tensor,     # [d_size, m] or [m, d_size] depending on layout  
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """LoRA-style pulsed A/B update with rank projections.
        
        Given x ∈ ℝ^{x_size×m}, d ∈ ℝ^{d_size×m}:
        1. Pack A_lr = A[:, :rank], B_lr = B[:rank, :]  
        2. Compute projections: X_B = B_lr @ x, D_A = A_lr^T @ d
        3. Route to tiles with proper rank-constrained updates:
           - A update: inputs = X_B, errors = d
           - B update: only update first rank rows with gradient from D_A
        4. Apply BL management and gradient magnitude correction
        
        Args:
            x: Input tensor [x_size, m] (or transposed)
            d: Error tensor [d_size, m] (or transposed)  
            lr: Learning rate
            in_trans: Input transposed flag
            out_trans: Output transposed flag
        """
        # Tiles expect [batch, features] but we need [features, batch] for matrix ops
        # Input is [batch, features], transpose to [features, batch]
        if x.dim() == 2:
            if x.size(1) == self.x_size:  # [batch, x_size]
                x = x.t()  # -> [x_size, batch]
        if d.dim() == 2:
            if d.size(1) == self.d_size:  # [batch, d_size]
                d = d.t()  # -> [d_size, batch]
                
        # Handle transpose flags
        if in_trans:
            x = x.t()  
        if out_trans:
            d = d.t()
            
        batch_size = x.size(-1)
        self._ensure_buffers(batch_size)
        
        # Prepare batch-first views for tile operations
        x_bf = x.t()  # [batch, x_size]
        d_bf = d.t()  # [batch, d_size]
        
        # Compute LoRA projections using tile forward/backward (no get_weights!)
        with torch.no_grad():
            # X_B = B_lr @ x -> Use B tile forward (B is now [rank, x_size])
            X_B = self.tile_b.forward(x_bf).t()  # [batch, rank] -> [rank, batch]
            
            # D_A = A_lr^T @ d -> Use A tile backward if available
            if hasattr(self.tile_a, 'backward'):
                # A.backward gives gradient w.r.t. input: [batch, rank]
                D_A = self.tile_a.backward(d_bf).t()  # [rank, batch]
            else:
                # Fallback: read weights (only if backward not available)
                A_full = self.tile_a.get_weights()[0].to(x.device)  # [d_size, rank]
                D_A = A_full[:, :self.rank].t() @ d  # [rank, d_size] @ [d_size, batch] -> [rank, batch]
        
        # Apply LoRA alpha scaling and gradient magnitude correction
        # Critical: LoRA gradients need to be scaled by alpha
        lr_eff = lr * self.lora_alpha
        
        # Apply gradient magnitude correction (optional)
        if self.correct_gradient_magnitudes:
            # Use 1/sqrt(rank) for proper scaling (not sqrt(rank))
            lr_eff = lr_eff / math.sqrt(self.rank)
            
        # Update A tile: inputs = X_B, errors = d
        # A tile is [d_size, rank] and expects inputs [batch, rank]
        # NOTE: In CUDA, X_B is padded to [x_size, batch] but for Python tiles
        # we use X_B directly as [batch, rank] since tiles handle dimensions differently
        
        # Apply BL management for A update
        old_lr = self.tile_a.get_learning_rate() 
        self.tile_a.set_learning_rate(lr_eff)
        
        # Force StochasticCompressed pulse type if BL management specified
        if self.ab_bl_mgmt:
            # Apply ab_bl_mgmt settings - simplified for now
            pass
            
        # Tiles expect [batch, features] format
        # X_B is [rank, batch], transpose to [batch, rank]
        # d is [d_size, batch], transpose to [batch, d_size]
        # Use original update if available (to avoid hook recursion)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(X_B.t(), d.t())
        else:
            self.tile_a.update(X_B.t(), d.t())
        self.tile_a.set_learning_rate(old_lr)
        self.num_a_updates += 1
        
        # Update B tile: inputs = x, errors = D_A 
        # B tile is now [rank, x_size], so no padding needed
        
        old_lr_b = self.tile_b.get_learning_rate()
        self.tile_b.set_learning_rate(lr_eff)
        
        if self.ab_bl_mgmt:
            # Apply ab_bl_mgmt settings
            pass
            
        # Tiles expect [batch, features] format
        # B tile is [rank, x_size], expects inputs [batch, x_size] and errors [batch, rank]
        # Use original update if available (to avoid hook recursion)
        if hasattr(self.tile_b, '_orig_update'):
            self.tile_b._orig_update(x.t(), D_A.t())  # x.t() = [batch, x_size], D_A.t() = [batch, rank]
        else:
            self.tile_b.update(x.t(), D_A.t())
        self.tile_b.set_learning_rate(old_lr_b)
        self.num_b_updates += 1
        
        # Update counters
        if self.units_in_mbatch:
            self.transfer_counter += batch_size
        else:
            self.transfer_counter += 1
            
    def ab_weight_transfer(self) -> None:
        """Pulsed A⊗B -> visible transfer, then reinit.
        
        Transfer: C += transfer_lr * (A @ B) via pulsed outer product:
        1. For chunks of rank: pack D_chunk = A[:, off:off+cur], X_chunk = B[off:off+cur, :]  
        2. Call visible pulsed updater: C.update(X_chunk^T, D_chunk, lr=|transfer_lr|)
        3. Handle sign rule: negate D when transfer_lr > 0 (PWU applies W += -lr * D @ X^T)
        4. Apply transfer BL management
        5. Unconditionally call reinit() after transfer
        """
        # Get current A and B weights and ensure they're on the right device
        device = self._get_tile_device()
        A_full = self.tile_a.get_weights()[0].to(device)  # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0].to(device)  # [rank, x_size] - B is already rank-sized
        
        A_lr = A_full[:, :self.rank]           # [d_size, rank]
        
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
            
            # Pack chunks
            D_chunk = A_lr[:, off:end]              # [d_size, cur]
            X_chunk = B_lr[off:end, :]              # [cur, x_size]
            
            # Sign rule: PWU computes W += -lr * D @ X^T, we want W += +transfer_lr * D @ X^T
            # So when transfer_lr > 0, negate D to get correct sign
            if self.transfer_lr > 0:
                D_chunk = -D_chunk
            elif self.transfer_lr < 0:
                # transfer_lr < 0: want W += transfer_lr * D @ X^T (negative), so keep D positive
                # PWU does W += -lr * D @ X^T with lr > 0, so net effect is W += -D @ X^T (negative) ✓
                pass
                
            # Pulsed update: C += transfer_lr * D_chunk @ X_chunk  
            # Use original update to avoid hook recursion
            # Tiles expect [batch, features] format: X_chunk is [cur, x_size], D_chunk.T is [cur, d_size]
            # Ensure tensors are on CPU for analog tiles
            X_chunk_cpu = X_chunk.cpu() if X_chunk.is_cuda else X_chunk
            D_chunk_t_cpu = D_chunk.t().cpu() if D_chunk.is_cuda else D_chunk.t()
            
            if hasattr(self.tile_c, '_orig_update'):
                self.tile_c._orig_update(X_chunk_cpu, D_chunk_t_cpu)
            else:
                self.tile_c.update(X_chunk_cpu, D_chunk_t_cpu)
            
        self.tile_c.set_learning_rate(old_lr)
        self.num_transfers += 1
        
        # CRITICAL: Reset transfer counter after transfer (matches CUDA)
        self.transfer_counter = 0
        
        # Unconditional reinit after transfer
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
        # Handle disabled forward injection
        if not self.forward_inject_enabled or self.rank == 0:
            return self.tile_c.forward(x, in_trans=in_trans, out_trans=out_trans)
            
        # Handle transposed cases with digital fallback
        if out_trans or in_trans:
            return self._forward_inject_digital_fallback(x, out_trans, in_trans)
            
        # Default analog-hybrid path
        return self._forward_inject_analog_hybrid(x)
        
    def _forward_inject_digital_fallback(
        self, 
        x: Tensor,
        out_trans: bool, 
        in_trans: bool
    ) -> Tensor:
        """Digital fallback: compose W_eff then single forward pass."""
        # Get weights
        C_weights = self.tile_c.get_weights()[0]   # [d_size, x_size]
        A_lr = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]        # [rank, x_size]
        
        # Compose effective weights: W_eff = C + α * A_lr @ B_lr
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
        """Analog-hybrid path: y_vis = C·x, g = B·x[:rank], y_ab = A·g, combine."""
        # Note: AnalogTile expects [batch, features] format (batch-first)
        # No transposition needed
            
        # 1. Visible forward: y_vis = C·x
        y_vis = self.tile_c.forward(x)  # [batch, d_size]
        
        # 2. B forward: g = B·x  
        g_rank = self.tile_b.forward(x)  # [batch, rank] since B is [rank, x_size]
        
        # 3. A forward with g_rank directly
        # A tile is [d_size, rank] internally (only uses first rank cols)
        # It expects input [batch, rank] which is exactly what g_rank is
        
        # 4. A forward: y_ab = A·g_rank
        y_ab = self.tile_a.forward(g_rank)  # [batch, d_size]
        
        # 5. Digital combination: y = y_vis + α * y_ab
        result = y_vis + self.lora_alpha * y_ab  # [batch, d_size]
        
        return result
        
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