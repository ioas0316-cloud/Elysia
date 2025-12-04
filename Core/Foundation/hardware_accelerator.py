import torch
import logging

class HardwareAccelerator:
    """
    Manages hardware acceleration resources (CPU/GPU).
    Detects CUDA availability and provides methods to utilize the best available device.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = self._detect_device()
        self.logger.info(f"Hardware Accelerator initialized on: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            self._log_memory_stats()

    def _detect_device(self) -> torch.device:
        """
        Detects if CUDA is available and returns the appropriate torch device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def get_device(self) -> torch.device:
        """
        Returns the currently active torch device.
        """
        return self.device

    def tensor(self, data, dtype=None) -> torch.Tensor:
        """
        Creates a tensor on the active device.
        """
        return torch.tensor(data, device=self.device, dtype=dtype)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Moves a tensor to the active device.
        """
        return tensor.to(self.device)

    def get_memory_stats(self) -> dict:
        """
        Returns memory statistics for the current device.
        Returns empty dict if on CPU.
        """
        stats = {}
        if self.device.type == 'cuda':
            stats['allocated'] = torch.cuda.memory_allocated(0)
            stats['reserved'] = torch.cuda.memory_reserved(0)
            stats['max_allocated'] = torch.cuda.max_memory_allocated(0)
        return stats

    def _log_memory_stats(self):
        """
        Logs current memory statistics.
        """
        if self.device.type == 'cuda':
            stats = self.get_memory_stats()
            self.logger.info(
                f"VRAM Stats - Allocated: {stats['allocated'] / 1024**2:.2f} MB, "
                f"Reserved: {stats['reserved'] / 1024**2:.2f} MB"
            )

# Global instance for easy access
accelerator = HardwareAccelerator()
