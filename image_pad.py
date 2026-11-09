"""
Self-contained padding helper for MagikImageResize.

Supports four pad styles:
  - "color"          solid color fill
  - "edge"            replicate the border pixels outward
  - "edge_pixel"      same as "edge" (kept as a distinct option for parity
                       with the original combo list / older workflows)
  - "pillarbox_blur"  replicate + blur the border for a soft background,
                       then paste the sharp original back on top
"""

import torch
import torch.nn.functional as F


class ImagePad:
    @staticmethod
    def pad(image, left, right, top, bottom, extra_padding=0, color="0, 0, 0", mode="color", mask=None):
        """
        image: (B, H, W, C) float tensor in [0, 1]
        mask:  (B, H, W) float tensor, or None
        Returns (image, mask) with mask staying None if it came in as None.
        """
        left, right, top, bottom = (
            max(left + extra_padding, 0),
            max(right + extra_padding, 0),
            max(top + extra_padding, 0),
            max(bottom + extra_padding, 0),
        )

        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return image, mask

        img_bchw = image.movedim(-1, 1).contiguous()
        B, C, H, W = img_bchw.shape
        new_h, new_w = H + top + bottom, W + left + right

        if mode in ("edge", "edge_pixel"):
            padded = F.pad(img_bchw, (left, right, top, bottom), mode="replicate")

        elif mode == "pillarbox_blur":
            background = F.pad(img_bchw, (left, right, top, bottom), mode="replicate")
            background = ImagePad._box_blur(background, radius=max(4, min(new_w, new_h) // 32))
            background[:, :, top:top + H, left:left + W] = img_bchw
            padded = background

        else:  # "color" (also the fallback for any unrecognized mode)
            rgb = ImagePad.parse_color(color, C)
            padded = img_bchw.new_zeros((B, C, new_h, new_w))
            for c in range(C):
                padded[:, c, :, :] = rgb[c]
            padded[:, :, top:top + H, left:left + W] = img_bchw

        out_image = padded.movedim(1, -1)

        out_mask = None
        if mask is not None:
            out_mask = F.pad(mask.unsqueeze(1), (left, right, top, bottom), mode="constant", value=0.0).squeeze(1)

        return out_image, out_mask

    @staticmethod
    def parse_color(color, channels):
        """Parse a '0-255' comma-separated color string into normalized [0,1] floats."""
        try:
            parts = [float(p.strip()) / 255.0 for p in color.split(",") if p.strip() != ""]
        except ValueError:
            parts = []
        if not parts:
            parts = [0.0, 0.0, 0.0]
        while len(parts) < 3:
            parts.append(parts[-1])
        if channels == 4 and len(parts) < 4:
            parts.append(1.0)  # opaque alpha by default
        return [max(0.0, min(1.0, v)) for v in parts[:channels]]

    @staticmethod
    def _box_blur(image_bchw, radius=8):
        """Cheap separable box blur, no external deps."""
        if radius <= 0:
            return image_bchw
        k = radius * 2 + 1
        channels = image_bchw.shape[1]
        weight_h = image_bchw.new_ones((channels, 1, 1, k)) / k
        weight_v = image_bchw.new_ones((channels, 1, k, 1)) / k
        blurred = F.conv2d(image_bchw, weight_h, padding=(0, radius), groups=channels)
        blurred = F.conv2d(blurred, weight_v, padding=(radius, 0), groups=channels)
        return blurred
