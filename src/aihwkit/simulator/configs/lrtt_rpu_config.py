"""Python LRTT RPU configuration for GPU version."""

from dataclasses import dataclass
from typing import Type

from aihwkit.simulator.configs.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


@dataclass
class PythonLRTTRPUConfig(SingleRPUConfig):
    """Configuration for Python-based LRTT RPU.
    
    This configuration uses the LRTTSimulatorTile implementation which
    orchestrates three analog tiles for Low-Rank Tensor-Train training.
    Works with both CPU and GPU versions of aihwkit.
    """
    
    tile_class: Type = LRTTSimulatorTile
    """Tile class that corresponds to this RPUConfig."""
    
    device: PythonLRTTDevice = None
    """Device parameters for LRTT."""
    
    def __post_init__(self):
        """Initialize default device if not provided."""
        if self.device is None:
            self.device = PythonLRTTDevice()
    
    def get_brief_info(self) -> str:
        """Get brief configuration info."""
        return (f"PythonLRTTRPUConfig(rank={self.device.rank}, "
                f"transfer_every={self.device.transfer_every}, "
                f"lora_alpha={self.device.lora_alpha})")