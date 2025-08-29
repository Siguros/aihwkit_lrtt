# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Python-level LRTT configuration classes.

Pure Python LRTT configurations designed specifically for our Python-level
implementation, eliminating CUDA dependencies and rpucuda bindings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import warnings

from aihwkit.simulator.configs.devices import PulsedDevice, ConstantStepDevice
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.parameters.helpers import _PrintableMixin


@dataclass  
class PythonLRTTDevice(_PrintableMixin):
    """Python-level LRTT device configuration.
    
    Designed specifically for Python LRTT implementation without CUDA dependencies.
    Maps directly to LRTTController parameters.
    """
    
    # === Core LRTT Parameters ===
    rank: int = 4
    """LoRA rank dimension r. Must be > 0 and <= min(d_size, x_size)."""
    
    transfer_every: int = 32
    """Transfer frequency: every N steps (or samples if units_in_mbatch=True)."""
    
    transfer_lr: float = 1.0
    """Transfer learning rate scalar applied during A⊗B -> visible transfer."""
    
    lora_alpha: float = 1.0
    """LoRA scaling factor α in W_eff = W_visible + α * A @ B."""
    
    reinit_gain: float = 0.1
    """Kaiming initialization gain for B matrix after transfer."""
    
    # === Advanced Parameters ===
    units_in_mbatch: bool = False
    """If True, transfer_every counts samples; if False, counts steps."""
    
    correct_gradient_magnitudes: bool = False
    """If True, scale learning rate by sqrt(rank) for gradient correction."""
    
    forward_inject: bool = True
    """Enable forward injection optimization: W_eff composition."""
    
    rank_chunk: Optional[int] = None
    """Chunk size for transfer (None = use full rank). For memory management."""
    
    columns_mode: bool = True
    """Transfer mode: True=columns (forward), False=rows (backward)."""
    
    # === Device Configuration ===
    unit_cell_devices: List[PulsedDevice] = field(default_factory=lambda: [
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0),
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0), 
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0)
    ])
    """Device configurations for [fastA, fastB, visible] tiles."""
    
    # === BL Management (Simplified for Python) ===
    ab_bl_mgmt: Optional[Dict[str, Any]] = None
    """BL management settings for A/B updates (optional)."""
    
    transfer_bl_mgmt: Optional[Dict[str, Any]] = None  
    """BL management settings for transfers (optional)."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate rank
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
            
        # Validate transfer parameters
        if self.transfer_every <= 0:
            raise ValueError(f"transfer_every must be positive, got {self.transfer_every}")
            
        if self.transfer_lr <= 0:
            raise ValueError(f"transfer_lr must be positive, got {self.transfer_lr}")
            
        # Validate LoRA parameters
        if self.lora_alpha < 0:
            raise ValueError(f"lora_alpha must be non-negative, got {self.lora_alpha}")
            
        if self.reinit_gain < 0:
            raise ValueError(f"reinit_gain must be non-negative, got {self.reinit_gain}")
            
        # Validate rank_chunk
        if self.rank_chunk is not None and self.rank_chunk <= 0:
            raise ValueError(f"rank_chunk must be positive or None, got {self.rank_chunk}")
            
        # Validate unit cell devices
        if len(self.unit_cell_devices) != 3:
            raise ValueError(f"Must provide exactly 3 unit_cell_devices for [fastA, fastB, visible], got {len(self.unit_cell_devices)}")
            
        # Set default rank_chunk
        if self.rank_chunk is None:
            self.rank_chunk = self.rank
            
        # Initialize BL management if not provided
        if self.ab_bl_mgmt is None:
            self.ab_bl_mgmt = {}
        if self.transfer_bl_mgmt is None:
            self.transfer_bl_mgmt = {}
    
    def get_device_for_tile(self, tile_type: str) -> PulsedDevice:
        """Get device configuration for specific tile type.
        
        Args:
            tile_type: 'fastA', 'fastB', or 'visible'
            
        Returns:
            Device configuration for the specified tile
        """
        tile_map = {'fastA': 0, 'fastB': 1, 'visible': 2}
        if tile_type not in tile_map:
            raise ValueError(f"Unknown tile_type '{tile_type}', must be one of {list(tile_map.keys())}")
            
        return self.unit_cell_devices[tile_map[tile_type]]
    
    def to_controller_kwargs(self) -> Dict[str, Any]:
        """Convert to LRTTController constructor arguments.
        
        Returns:
            Dictionary of arguments for LRTTController.__init__()
        """
        return {
            'transfer_lr': self.transfer_lr,
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'correct_gradient_magnitudes': self.correct_gradient_magnitudes,
            'rank_chunk': self.rank_chunk,
            'ab_bl_mgmt': self.ab_bl_mgmt,
            'transfer_bl_mgmt': self.transfer_bl_mgmt,
            'forward_inject': self.forward_inject
        }
    
    @classmethod
    def from_legacy_lrtt_compound(cls, legacy_compound) -> 'PythonLRTTDevice':
        """Create from legacy LRTTTransferCompound for migration.
        
        Args:
            legacy_compound: LRTTTransferCompound instance
            
        Returns:
            Equivalent PythonLRTTDevice
        """
        return cls(
            rank=getattr(legacy_compound, 'rank', 4),
            transfer_every=getattr(legacy_compound, 'transfer_every', 32),
            transfer_lr=getattr(legacy_compound, 'transfer_lr', 1.0),
            lora_alpha=getattr(legacy_compound, 'lora_alpha', 1.0),
            reinit_gain=getattr(legacy_compound, 'reinit_gain', 0.1),
            units_in_mbatch=getattr(legacy_compound, 'units_in_mbatch', False),
            correct_gradient_magnitudes=getattr(legacy_compound, 'correct_gradient_magnitudes', False),
            forward_inject=getattr(legacy_compound, 'forward_inject', True),
            rank_chunk=getattr(legacy_compound, 'rank_chunk', None),
            unit_cell_devices=getattr(legacy_compound, 'unit_cell_devices', [ConstantStepDevice()] * 3)
        )


@dataclass
class PythonLRTTPreset(_PrintableMixin):
    """Preset configurations for common LRTT use cases."""
    
    @staticmethod
    def idealized(rank: int = 4, transfer_every: int = 32, lora_alpha: float = 1.0) -> PythonLRTTDevice:
        """Idealized LRTT with minimal noise and perfect devices.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            lora_alpha: LoRA scaling factor
            
        Returns:
            Idealized PythonLRTTDevice configuration
        """
        from aihwkit.simulator.configs.devices import IdealDevice
        
        ideal_device = IdealDevice()
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.1,
            forward_inject=True,
            unit_cell_devices=[ideal_device, ideal_device, ideal_device]
        )
    
    @staticmethod
    def constant_step(rank: int = 4, transfer_every: int = 32, dw_min: float = 0.01) -> PythonLRTTDevice:
        """LRTT with ConstantStepDevice for all tiles.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency  
            dw_min: Minimum weight update step
            
        Returns:
            ConstantStep PythonLRTTDevice configuration
        """
        device = ConstantStepDevice(
            dw_min=dw_min,
            dw_min_dtod=0.0,
            up_down_dtod=0.0,
            w_min=-1.0,
            w_max=1.0
        )
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0,
            reinit_gain=0.1,
            forward_inject=True,
            unit_cell_devices=[device, device, device]
        )
    
    @staticmethod  
    def lora_style(rank: int = 8, lora_alpha: float = 16.0, transfer_every: int = 1) -> PythonLRTTDevice:
        """LoRA-style configuration with frequent transfers.
        
        Similar to standard LoRA but with analog tiles and periodic consolidation.
        
        Args:
            rank: LoRA rank (typically higher for LoRA-style)
            lora_alpha: LoRA alpha (typically higher: α = 16, 32)
            transfer_every: Transfer frequency (1 = every step)
            
        Returns:
            LoRA-style PythonLRTTDevice configuration
        """
        from aihwkit.simulator.configs.devices import IdealDevice
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.05,  # Smaller reinit for frequent transfers
            forward_inject=True,
            correct_gradient_magnitudes=True,  # Better scaling for higher ranks
            unit_cell_devices=[IdealDevice(), IdealDevice(), IdealDevice()]
        )
    
    @staticmethod
    def mixed_precision(rank: int = 4, transfer_every: int = 16) -> PythonLRTTDevice:
        """Mixed precision: high precision visible, lower precision A/B.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            
        Returns:
            Mixed precision PythonLRTTDevice configuration
        """
        # Lower precision for A/B (faster updates)
        low_precision = ConstantStepDevice(dw_min=0.05, w_min=-0.8, w_max=0.8)
        
        # Higher precision for visible (stable storage)
        high_precision = ConstantStepDevice(dw_min=0.001, w_min=-2.0, w_max=2.0)
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0,
            reinit_gain=0.1,
            forward_inject=True,
            unit_cell_devices=[low_precision, low_precision, high_precision]
        )
    
    @staticmethod
    def inference_optimized(rank: int = 2, lora_alpha: float = 0.5) -> PythonLRTTDevice:
        """Inference-optimized configuration with forward injection.
        
        Args:
            rank: Lower rank for faster inference
            lora_alpha: Lower alpha for stability
            
        Returns:
            Inference-optimized PythonLRTTDevice configuration
        """
        from aihwkit.simulator.configs.devices import IdealDevice
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=1,  # Transfer immediately
            lora_alpha=lora_alpha,
            reinit_gain=0.0,  # No reinit needed for inference
            forward_inject=True,  # Essential for inference
            columns_mode=True,  # Optimized mode
            unit_cell_devices=[IdealDevice(), IdealDevice(), IdealDevice()]
        )