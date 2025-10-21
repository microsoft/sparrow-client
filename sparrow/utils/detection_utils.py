"""Detection utilities: box scaling, clipping, and NMS helpers."""

from __future__ import annotations
import time
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from torchvision.ops import nms as tv_nms
from torchvision.ops import batched_nms as tv_batched_nms

ArrayLike = Union[np.ndarray, Tensor]


def _is_tensor(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def clip_boxes(boxes: ArrayLike, shape: Sequence[int]) -> ArrayLike:
    """
    Clip xyxy boxes to image shape (h, w).
    Args:
        boxes: (..., 4) boxes in xyxy
        shape: (h, w) of target image
    """
    h, w = int(shape[0]), int(shape[1])
    if _is_tensor(boxes):
        boxes[..., 0] = boxes[..., 0].clamp(0, w)  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, h)  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, w)  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, h)  # y2
    else:
        boxes[..., 0] = np.clip(boxes[..., 0], 0, w)
        boxes[..., 1] = np.clip(boxes[..., 1], 0, h)
        boxes[..., 2] = np.clip(boxes[..., 2], 0, w)
        boxes[..., 3] = np.clip(boxes[..., 3], 0, h)
    return boxes


def xywh2xyxy(x: ArrayLike) -> ArrayLike:
    """
    Convert (cx, cy, w, h) -> (x1, y1, x2, y2).
    """
    if _is_tensor(x):
        y = torch.empty_like(x)
    else:
        y = np.empty_like(x)

    cxcy = x[..., :2]
    wh2 = x[..., 2:] * 0.5
    y[..., :2] = cxcy - wh2
    y[..., 2:] = cxcy + wh2
    return y


def scale_boxes(
    img1_shape: Sequence[int],
    boxes: ArrayLike,
    img0_shape: Sequence[int],
    ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> ArrayLike:
    """
    Rescale boxes from a letterboxed/canvas image (img1_shape) back to the
    original image size (img0_shape). Boxes are xyxy.

    If ratio_pad is provided, it should be ((gain_w, gain_h), (pad_w, pad_h))
    in pixels relative to img1_shape. Otherwise we estimate from shapes.

    Args:
        img1_shape: (h1, w1) of network input
        boxes: (..., 4) in xyxy coords w.r.t img1
        img0_shape: (h0, w0) of original image (or array with .shape)
        ratio_pad: optional ((gain_w, gain_h), (pad_w, pad_h))
    """
    if hasattr(img0_shape, "shape"):
        img0_h, img0_w = int(img0_shape.shape[0]), int(img0_shape.shape[1])
    else:
        img0_h, img0_w = int(img0_shape[0]), int(img0_shape[1])
    img1_h, img1_w = int(img1_shape[0]), int(img1_shape[1])

    if ratio_pad is None:
        gain = min(img1_w / img0_w, img1_h / img0_h)  # uniform gain used during letterboxing
        pad_w = (img1_w - img0_w * gain) * 0.5
        pad_h = (img1_h - img0_h * gain) * 0.5
    else:
        (gain_w, gain_h), (pad_w, pad_h) = ratio_pad
        # If non-uniform gain is passed, fall back to average to keep behavior predictable
        gain = float(gain_w + gain_h) * 0.5

    if _is_tensor(boxes):
        boxes = boxes.clone()
        boxes[..., [0, 2]] -= pad_w
        boxes[..., [1, 3]] -= pad_h
        boxes[..., :4] /= gain
    else:
        boxes = boxes.copy()
        boxes[..., [0, 2]] = (boxes[..., [0, 2]] - pad_w) / gain
        boxes[..., [1, 3]] = (boxes[..., [1, 3]] - pad_h) / gain

    return clip_boxes(boxes, (img0_h, img0_w))


@torch.no_grad()
def non_max_suppression(
    prediction: Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[Sequence[int]] = None,
    agnostic: bool = False,
    max_det: int = 300,
    max_time_img: float = 0.05,
) -> List[Tensor]:
    """
    Generic NMS that supports multiple common formats and **auto-detects layout**:

    Accepted layouts:
      • (B, N, C)  candidates-first
      • (B, C, N)  channels-first  <-- will be auto-transposed to (B, N, C)

    Accepted channel formats (C):
      1) C == 6: (x1, y1, x2, y2, conf, cls)
      2) C >= 6: (cx, cy, w, h) + per-class scores (K = C - 4)

    Returns:
        List of length B; each element is a Tensor of shape [M, 6]:
        (x1, y1, x2, y2, conf, cls)
    """
    assert prediction.ndim == 3, f"Expected 3D tensor, got {prediction.shape}"
    B, D1, D2 = prediction.shape

    # Auto-detect layout: channels should be small (<= ~16), candidates large (>= ~50)
    if D1 <= 16 and D2 >= 50:
        # Probably (B, C, N) -> transpose to (B, N, C)
        prediction = prediction.transpose(1, 2)
    elif D2 <= 16 and D1 >= 50:
        # Already (B, N, C)
        pass
    else:
        # Ambiguous; assume (B, N, C)
        pass

    B, N, C = prediction.shape
    t_start = time.time()
    outputs: List[Tensor] = []

    for b in range(B):
        x = prediction[b]

        # Case 1: already in (xyxy, conf, cls)
        if C == 6:
            det = x
            if classes is not None:
                allowed = torch.as_tensor(classes, device=det.device).float()
                det = det[(det[:, 5:6] == allowed).any(1)]
            det = det[det[:, 4] >= conf_thres]
            if det.numel() == 0:
                outputs.append(det)
                continue

            if agnostic:
                keep = tv_nms(det[:, :4], det[:, 4], iou_thres)
            else:
                keep = tv_batched_nms(det[:, :4], det[:, 4], det[:, 5], iou_thres)
            outputs.append(det[keep][:max_det])
            continue

        # Case 2: (cx,cy,w,h) + K class scores
        if C < 6:
            # Not enough channels for boxes + at least 2 classes; return empty
            outputs.append(torch.zeros((0, 6), device=x.device))
            continue

        boxes_xyxy = xywh2xyxy(x[:, :4])
        scores_per_class = x[:, 4:]  # [N, K]
        K = scores_per_class.shape[1]

        # Best class per box
        cls_conf, cls_idx = scores_per_class.max(dim=1)
        det_mask = cls_conf >= conf_thres

        if classes is not None:
            allowed = torch.as_tensor(classes, device=x.device)
            det_mask = det_mask & ((cls_idx[:, None] == allowed[None, :]).any(dim=1))

        boxes_xyxy = boxes_xyxy[det_mask]
        cls_conf = cls_conf[det_mask]
        cls_idx = cls_idx[det_mask]

        if boxes_xyxy.numel() == 0:
            outputs.append(torch.zeros((0, 6), device=x.device))
            continue

        # NMS (batched by class unless agnostic)
        if agnostic:
            keep = tv_nms(boxes_xyxy, cls_conf, iou_thres)
        else:
            keep = tv_batched_nms(boxes_xyxy, cls_conf, cls_idx, iou_thres)

        keep = keep[:max_det]
        det = torch.cat(
            (boxes_xyxy[keep], cls_conf[keep, None], cls_idx[keep].float().unsqueeze(1)),
            dim=1,
        )  # [M, 6]
        outputs.append(det)

        # Safety time limit per image
        if (time.time() - t_start) > (2.0 + max_time_img):
            break

    return outputs
