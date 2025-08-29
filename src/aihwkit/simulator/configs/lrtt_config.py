# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Python-level LRTT RPU configuration.

RPU configuration classes designed specifically for Python LRTT implementation.
"""

from dataclasses import dataclass, field
from typing import Type, Any

from aihwkit.simulator.configs.configs import IOManagedRPUConfig
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.enums import RPUDataType


@dataclass
class PythonLRTTRPUConfig(IOManagedRPUConfig):
    """RPU Configuration for Python-level LRTT implementation.
    
    This configuration automatically selects LRTTSimulatorTile when used
    and provides all necessary parameters for Python LRTT operation.
    """
    
    tile_class: Type = LRTTSimulatorTile
    """Tile class: Always LRTTSimulatorTile for Python LRTT."""
    
    tile_array_class: Type = TileModuleArray
    """Tile array class for multi-tile scenarios."""
    
    device: PythonLRTTDevice = field(default_factory=PythonLRTTDevice)
    """Python LRTT device configuration."""
    
    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Always returns LRTTSimulatorTile for Python LRTT."""
        return LRTTSimulatorTile
    
    def create_tile(self, x_size: int, d_size: int, dtype: RPUDataType = RPUDataType.FLOAT, **kwargs) -> LRTTSimulatorTile:
        """Create LRTTSimulatorTile with this configuration.
        
        Args:
            x_size: Input size
            d_size: Output size  
            dtype: Data type for tiles
            **kwargs: Additional arguments
            
        Returns:
            Configured LRTTSimulatorTile instance
        """
        return LRTTSimulatorTile(
            x_size=x_size,
            d_size=d_size,
            rpu_config=self,
            dtype=dtype,
            **kwargs
        )
    
    def validate_dimensions(self, d_size: int, x_size: int) -> None:
        """Validate that dimensions are compatible with rank.
        
        Args:
            d_size: Output dimension
            x_size: Input dimension
            
        Raises:
            ValueError: If rank is incompatible with dimensions
        """
        if self.device.rank > min(d_size, x_size):
            raise ValueError(
                f"Rank {self.device.rank} too large for dimensions {d_size}×{x_size}. "
                f"Maximum rank is {min(d_size, x_size)}"
            )
    
    def get_brief_info(self) -> str:
        """Get brief configuration info."""
        return (f"PythonLRTTRPUConfig(rank={self.device.rank}, "
                f"transfer_every={self.device.transfer_every}, "
                f"lora_alpha={self.device.lora_alpha})")
    
    def as_bindings(self) -> Any:
        """Return self - no CUDA bindings needed for Python implementation.""" 
        return self


# Convenience factory functions for common configurations

def lrtt_idealized_config(rank: int = 4, transfer_every: int = 32, lora_alpha: float = 1.0) -> PythonLRTTRPUConfig:
    """Create idealized LRTT configuration.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency  
        lora_alpha: LoRA scaling factor
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.idealized(rank=rank, transfer_every=transfer_every, lora_alpha=lora_alpha)
    return PythonLRTTRPUConfig(device=device)


def lrtt_constant_step_config(rank: int = 4, transfer_every: int = 32, dw_min: float = 0.01) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with ConstantStepDevice.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency
        dw_min: Minimum weight update step
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.constant_step(rank=rank, transfer_every=transfer_every, dw_min=dw_min)
    return PythonLRTTRPUConfig(device=device)


def lrtt_lora_style_config(rank: int = 8, lora_alpha: float = 16.0, transfer_every: int = 1) -> PythonLRTTRPUConfig:
    """Create LoRA-style LRTT configuration.
    
    Args:
        rank: LoRA rank (higher for LoRA-style)
        lora_alpha: LoRA alpha (higher for LoRA-style)
        transfer_every: Transfer frequency (1 = every step)
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.lora_style(rank=rank, lora_alpha=lora_alpha, transfer_every=transfer_every)
    return PythonLRTTRPUConfig(device=device)


def lrtt_mixed_precision_config(rank: int = 4, transfer_every: int = 16) -> PythonLRTTRPUConfig:
    """Create mixed precision LRTT configuration.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.mixed_precision(rank=rank, transfer_every=transfer_every)
    return PythonLRTTRPUConfig(device=device)


def lrtt_inference_config(rank: int = 2, lora_alpha: float = 0.5) -> PythonLRTTRPUConfig:
    """Create inference-optimized LRTT configuration.
    
    Args:
        rank: Lower rank for inference
        lora_alpha: Lower alpha for stability
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.inference_optimized(rank=rank, lora_alpha=lora_alpha)
    return PythonLRTTRPUConfig(device=device)


# Legacy compatibility layer
def migrate_from_legacy_lrtt_compound(legacy_compound) -> PythonLRTTRPUConfig:
    """Migrate from legacy LRTTTransferCompound to Python LRTT.
    
    Args:
        legacy_compound: LRTTTransferCompound instance
        
    Returns:
        Equivalent PythonLRTTRPUConfig
    """
    python_device = PythonLRTTDevice.from_legacy_lrtt_compound(legacy_compound)
    return PythonLRTTRPUConfig(device=python_device)