"""
Camera Interface (Hardware-Agnostic)
=====================================
Abstract base class for all image and video capture inputs.
Allows seamless swapping between laptop webcam, smartphone IP stream,
Raspberry Pi CSI camera, or future smart glasses hardware without
modifying upstream perception or intelligence modules.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseCamera(ABC):
    """
    Abstract contract for camera devices.
    """

    @abstractmethod
    def open(self) -> bool:
        """Initialize and open the camera connection."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame.
        
        Returns:
            (success, frame): Boolean status and BGR numpy array (H, W, 3).
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release underlying hardware or network stream resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera stream is active and readable."""
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) resolution of the capture device."""
        pass
