"""Utilities to resize/pad/truncate spectrogram tensors to (H, W). Expects [C, H, W]."""

from typing import List, Optional
import torch
import torch.nn.functional as F

class ResizeTo:
    """Resize/pad/truncate a spectrogram tensor to (H, W). Expects [C,H,W]."""
    def __init__(self, size_hw: List[int], pad_value: Optional[float] = -80.0):
        self.size_hw = size_hw
        self.pad_value = pad_value  # e.g. -80.0 for dB; None -> use x.min()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ResizeTo expects [C,H,W], got {tuple(x.shape)}")

        target_h, target_w = self.size_hw
        C, H, W = x.shape

        if H != target_h:
            x = F.interpolate(x.unsqueeze(0), size=(target_h, W),
                              mode="bilinear", align_corners=False).squeeze(0)

        if W < target_w:
            pad_val = self.pad_value if self.pad_value is not None else float(x.min().item())
            pad_right = target_w - W
            x = F.pad(x, (0, pad_right, 0, 0), mode="constant", value=pad_val)
        elif W > target_w:
            x = x[:, :, :target_w]

        return x
