"""CPU fallback for SAM2's connected-components op.

SAM2's own version is a CUDA extension (sam2._C) that needs nvcc to build. We skip building it
(see app.py), so this pure OpenCV version stands in. It is only used for SAM2's "fill small holes in
the mask" step, so it must match the contract: for mask (N,1,H,W) return
  labels - component id per foreground pixel (0 for background)
  counts - area of the pixel's component (0 for background)
Connectivity is 4 (what the CUDA version uses).
"""
import cv2
import numpy as np
import torch


def get_connected_components_cpu(mask: torch.Tensor):
    m = mask.to(torch.uint8).cpu().numpy()
    n = m.shape[0]
    labels = np.zeros(m.shape, dtype=np.int32)
    counts = np.zeros(m.shape, dtype=np.int32)
    for i in range(n):
        _, lab, stats, _ = cv2.connectedComponentsWithStats(m[i, 0], connectivity=4)
        labels[i, 0] = lab
        area = stats[:, cv2.CC_STAT_AREA].astype(np.int32)
        area[0] = 0  # background component
        counts[i, 0] = area[lab]
    return torch.from_numpy(labels).to(mask.device), torch.from_numpy(counts).to(mask.device)
