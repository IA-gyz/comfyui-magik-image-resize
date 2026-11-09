"""
MagikImageResize - standalone ComfyUI image resize node.

Resizes an image (and its accompanying mask, if any) to a target width/height
with a choice of upscale algorithm and aspect-ratio handling strategy
(stretch, resize-then-pad, crop-to-fit, or fit-by-total-pixel-count).

No external custom-node packages required - only core ComfyUI
(torch, comfy.utils, comfy.model_management) plus the standard library.
"""

import logging
import math

import torch

import comfy.model_management as model_management
from comfy.utils import common_upscale

try:
    from nodes import MAX_RESOLUTION
except ImportError:
    MAX_RESOLUTION = 16384

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from comfy_api.latest import io

from .image_pad import ImagePad


class MagikImageResize(io.ComfyNode):
    """Resize an IMAGE (and optional MASK) to an exact width/height."""

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]
    OUTPAINT_MODE = "pad for outpaint"
    MEGAPIXEL_MODE = "Target Pixel Budget (Megapixels)"
    KEEP_PROPORTION_MODES = [
        "stretch", "resize", "pad", "pad_edge", "pad_edge_pixel",
        OUTPAINT_MODE, "crop", "pillarbox_blur", MEGAPIXEL_MODE,
    ]
    MEGAPIXEL_OPTIONS = [
        "0.25 MP (~512x512)",
        "0.5 MP (~768x768)",
        "1.0 MP (~1024x1024)",
        "1.5 MP (~1280x1280)",
        "2.0 MP (~1440x1440)",
        "4.0 MP (~2048x2048)",
    ]
    CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]
    PAD_MODES = {"pad", "pad_edge", "pad_edge_pixel"}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Magik.ImageResize",
            display_name="Magik Image Resize",
            category="Magik/Image/Process",
            description="""
Resizes the image to the specified width and height.
Size can be retrieved from the input.

Keep proportions keeps the aspect ratio of the image, by
highest dimension.

"Target Pixel Budget (Megapixels)" uses the dedicated target_megapixels menu
to automatically scale the image to a total pixel budget while preserving its original aspect ratio.

"pad for outpaint" is different from the other modes: the image itself is
left at its original resolution (no scaling at all) and is instead placed
onto a new canvas of the requested width/height. Areas of the new canvas
not covered by the image are filled with pad_color (black by default) and
marked with a mask value of 1.0, ready to feed into an inpainting/outpainting
sampler. If the requested width/height is smaller than the image in a given
dimension, that dimension is cropped instead of padded.
""",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target width in pixels. Used for standard modes."),
                io.Int.Input("height", default=512, min=0, max=MAX_RESOLUTION, step=1, tooltip="Target height in pixels. Used for standard modes."),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS),
                io.Combo.Input("keep_proportion", options=cls.KEEP_PROPORTION_MODES, default="stretch", tooltip="Strategy for aspect ratio management. 'Target Pixel Budget' uses the megapixels dropdown below."),
                io.Combo.Input("target_megapixels", options=cls.MEGAPIXEL_OPTIONS, default="1.0 MP (~1024x1024)", tooltip="Target total pixel count when 'Target Pixel Budget' is selected."),
                io.String.Input("pad_color", default="0, 0, 0", tooltip="Color to use for padding."),
                io.Combo.Input("crop_position", options=cls.CROP_POSITIONS, default="center"),
                io.Int.Input("divisible_by", default=2, min=0, max=512, step=1),
                io.Mask.Input("mask", optional=True),
                io.Combo.Input("device", options=["cpu", "gpu"], default="cpu", optional=True),
                io.Int.Input(
                    "per_batch",
                    default=64, min=0, max=4096, step=1,
                    tooltip="Process the batch in chunks of this many images to reduce peak memory usage. 0 disables sub-batching (process everything at once).",
                    optional=True,
                ),
            ],
            outputs=[
                io.Image.Output("IMAGE", display_name="IMAGE"),
                io.Int.Output("width", display_name="width"),
                io.Int.Output("height", display_name="height"),
                io.Mask.Output("mask", display_name="mask"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _normalize_mask(mask, H, W):
        """Drop ComfyUI's 64x64 placeholder mask, and resize a real mask to match the image."""
        if mask is None:
            return None
        if mask.shape[-2:] == (64, 64) and (H != 64 or W != 64):
            return None
        if mask.shape[-2:] != (H, W):
            mask = common_upscale(mask.unsqueeze(1), W, H, "bilinear", crop="disabled").squeeze(1)
        return mask

    @staticmethod
    def _resolve_device(device, upscale_method):
        if device == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on the GPU")
            return model_management.get_torch_device()
        return torch.device("cpu")

    @classmethod
    def _compute_target_size(cls, W, H, width, height, keep_proportion, target_megapixels="1.0 MP (~1024x1024)"):
        """
        Work out the final (pre-padding) width/height given the source
        dimensions and the requested keep_proportion mode.
        """
        needs_fit = keep_proportion in ("resize", cls.MEGAPIXEL_MODE) or keep_proportion in cls.PAD_MODES or keep_proportion == "pillarbox_blur"

        if not needs_fit:
            new_width = width if width != 0 else W
            new_height = height if height != 0 else H
            return new_width, new_height

        if keep_proportion == cls.MEGAPIXEL_MODE:
            mp_map = {
                "0.25 MP (~512x512)": 512 * 512,
                "0.5 MP (~768x768)": 768 * 768,
                "1.0 MP (~1024x1024)": 1024 * 1024,
                "1.5 MP (~1280x1280)": 1280 * 1280,
                "2.0 MP (~1440x1440)": 1440 * 1440,
                "4.0 MP (~2048x2048)": 2048 * 2048,
            }
            total_pixels = mp_map.get(target_megapixels, 1024 * 1024)
            aspect_ratio = W / H
            new_height = int(math.sqrt(total_pixels / aspect_ratio))
            new_width = int(math.sqrt(total_pixels * aspect_ratio))
        elif width == 0 and height == 0:
            new_width, new_height = W, H
        elif width == 0:
            ratio = height / H
            new_width, new_height = round(W * ratio), height
        elif height == 0:
            ratio = width / W
            new_width, new_height = width, round(H * ratio)
        else:
            ratio = min(width / W, height / H)
            new_width, new_height = round(W * ratio), round(H * ratio)

        return new_width, new_height

    @staticmethod
    def _compute_padding(target_width, target_height, fit_width, fit_height, crop_position):
        """Padding needed to place a (fit_width x fit_height) image into a (target_width x target_height) canvas."""
        extra_w = target_width - fit_width
        extra_h = target_height - fit_height

        if crop_position == "top":
            top, bottom = 0, extra_h
            left = extra_w // 2
            right = extra_w - left
        elif crop_position == "bottom":
            top, bottom = extra_h, 0
            left = extra_w // 2
            right = extra_w - left
        elif crop_position == "left":
            left, right = 0, extra_w
            top = extra_h // 2
            bottom = extra_h - top
        elif crop_position == "right":
            left, right = extra_w, 0
            top = extra_h // 2
            bottom = extra_h - top
        else:  # center
            left = extra_w // 2
            right = extra_w - left
            top = extra_h // 2
            bottom = extra_h - top

        return left, right, top, bottom

    @staticmethod
    def _crop_to_aspect(tensor_image, tensor_mask, width, height, crop_position):
        """Center/edge-crop a (B,H,W,C) tensor to match the target aspect ratio."""
        old_height, old_width = tensor_image.shape[-3], tensor_image.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height

        if old_aspect > new_aspect:
            crop_w, crop_h = round(old_height * new_aspect), old_height
        else:
            crop_w, crop_h = old_width, round(old_width / new_aspect)

        if crop_position == "top":
            x, y = (old_width - crop_w) // 2, 0
        elif crop_position == "bottom":
            x, y = (old_width - crop_w) // 2, old_height - crop_h
        elif crop_position == "left":
            x, y = 0, (old_height - crop_h) // 2
        elif crop_position == "right":
            x, y = old_width - crop_w, (old_height - crop_h) // 2
        else:  # center
            x, y = (old_width - crop_w) // 2, (old_height - crop_h) // 2

        tensor_image = tensor_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
        if tensor_mask is not None:
            tensor_mask = tensor_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)
        return tensor_image, tensor_mask

    @staticmethod
    def _log_progress(message, unique_id):
        if unique_id and PromptServer is not None:
            try:
                PromptServer.instance.send_progress_text(message, unique_id)
                return
            except Exception:
                pass
        logging.info(f"[MagikImageResize] {message}")

    # -------------------------------------------------------------- resize

    @classmethod
    def execute(cls, image, width, height, keep_proportion, upscale_method, divisible_by,
                pad_color, crop_position, target_megapixels="1.0 MP (~1024x1024)", device="cpu", mask=None, per_batch=64):
        unique_id = cls.hidden.unique_id

        B, H, W, C = image.shape

        mask = cls._normalize_mask(mask, H, W)
        device = cls._resolve_device(device, upscale_method)

        if keep_proportion == cls.OUTPAINT_MODE:
            out_image, out_mask = cls._run_outpaint(
                image, mask, width, height, divisible_by, pad_color, crop_position, device, per_batch, unique_id,
            )
        else:
            out_image, out_mask = cls._run_standard(
                image, mask, width, height, keep_proportion, upscale_method, divisible_by,
                pad_color, crop_position, target_megapixels, device, per_batch, unique_id,
            )

        try:
            memory_size_mb = (out_image.numel() * out_image.element_size()) / (1024 * 1024)
            cls._log_progress(
                f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x "
                f"<b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                unique_id,
            )
        except Exception:
            pass

        final_mask = out_mask.cpu() if out_mask is not None else torch.zeros(
            1, 64, 64, device=torch.device("cpu"), dtype=torch.float32
        )
        return io.NodeOutput(out_image.cpu(), out_image.shape[2], out_image.shape[1], final_mask)

    # ------------------------------------------------------ standard resize

    @classmethod
    def _run_standard(cls, image, mask, width, height, keep_proportion, upscale_method, divisible_by,
                       pad_color, crop_position, target_megapixels, device, per_batch, unique_id):
        """Every keep_proportion mode except OUTPAINT_MODE: scales the image content to fit."""
        B, H, W, C = image.shape

        pillarbox_blur = keep_proportion == "pillarbox_blur"
        is_pad_mode = keep_proportion in cls.PAD_MODES or pillarbox_blur

        fit_width, fit_height = cls._compute_target_size(W, H, width, height, keep_proportion, target_megapixels)

        pad_left = pad_right = pad_top = pad_bottom = 0
        if is_pad_mode:
            pad_left, pad_right, pad_top, pad_bottom = cls._compute_padding(
                width, height, fit_width, fit_height, crop_position
            )

        width, height = fit_width, fit_height

        if divisible_by > 1:
            width -= width % divisible_by
            height -= height % divisible_by

        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()
                est_mb = (B * height * width * C * bytes_per_elem) / (1024 * 1024)
                cls._log_progress(
                    f"<tr><td>Resize</td><td>estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}</td></tr>",
                    unique_id,
                )
            except Exception:
                pass

        nvvfx_sr, nvvfx_ctx = cls._setup_rtx_vsr(upscale_method, width, height)

        def _process_subbatch(in_image, in_mask):
            out_image = in_image if in_image.device == device else in_image.to(device)
            out_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))

            if keep_proportion == "crop":
                out_image, out_mask = cls._crop_to_aspect(out_image, out_mask, width, height, crop_position)

            out_image, out_mask = cls._upscale(out_image, out_mask, width, height, upscale_method, nvvfx_sr)

            if is_pad_mode and (pad_left or pad_right or pad_top or pad_bottom):
                out_image, out_mask = cls._pad_result(
                    out_image, out_mask, pad_left, pad_right, pad_top, pad_bottom,
                    divisible_by, pad_color, keep_proportion, pillarbox_blur,
                )

            return out_image, out_mask

        if per_batch == 0 or B <= per_batch:
            out_image, out_mask = _process_subbatch(image, mask)
        else:
            out_image, out_mask = cls._process_in_chunks(image, mask, per_batch, _process_subbatch, unique_id)

        if nvvfx_ctx is not None:
            nvvfx_ctx.__exit__(None, None, None)

        return out_image, out_mask

    # --------------------------------------------------------- sub-helpers

    @staticmethod
    def _setup_rtx_vsr(upscale_method, width, height):
        if upscale_method != "nvidia_rtx_vsr":
            return None, None
        try:
            import nvvfx
        except ImportError:
            raise ImportError(
                "NVIDIA RTX Video Super Resolution is not available. "
                "Please install the nvidia-vfx library and ensure you have a compatible NVIDIA GPU."
            )
        nvvfx_ctx = nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)
        nvvfx_sr = nvvfx_ctx.__enter__()
        nvvfx_sr.output_width = max(8, round(width / 8) * 8)
        nvvfx_sr.output_height = max(8, round(height / 8) * 8)
        nvvfx_sr.load()
        return nvvfx_sr, nvvfx_ctx

    @staticmethod
    def _upscale(out_image, out_mask, width, height, upscale_method, nvvfx_sr):
        if upscale_method == "nvidia_rtx_vsr":
            frames_chw = out_image.movedim(-1, 1).cuda().contiguous()
            upscaled_frames = [
                torch.from_dlpack(nvvfx_sr.run(frames_chw[j]).image).clone()
                for j in range(frames_chw.shape[0])
            ]
            out_image = torch.stack(upscaled_frames, dim=0).movedim(1, -1).cpu()
            if out_mask is not None:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, "bilinear", crop="disabled").squeeze(1)
            return out_image, out_mask

        out_image = common_upscale(out_image.movedim(-1, 1), width, height, upscale_method, crop="disabled").movedim(1, -1)
        if out_mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(
                    out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled"
                ).movedim(1, -1)[:, :, :, 0]
            else:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)
        return out_image, out_mask

    @staticmethod
    def _pad_result(out_image, out_mask, pad_left, pad_right, pad_top, pad_bottom,
                     divisible_by, pad_color, keep_proportion, pillarbox_blur):
        width = out_image.shape[-2]
        height = out_image.shape[-3]
        padded_width = width + pad_left + pad_right
        padded_height = height + pad_top + pad_bottom

        if divisible_by > 1:
            width_remainder = padded_width % divisible_by
            height_remainder = padded_height % divisible_by
            if width_remainder > 0:
                pad_right += divisible_by - width_remainder
            if height_remainder > 0:
                pad_bottom += divisible_by - height_remainder

        pad_mode = (
            "pillarbox_blur" if pillarbox_blur else
            "edge" if keep_proportion == "pad_edge" else
            "edge_pixel" if keep_proportion == "pad_edge_pixel" else
            "color"
        )
        return ImagePad.pad(out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, pad_mode, mask=out_mask)

    @classmethod
    def _process_in_chunks(cls, image, mask, per_batch, process_fn, unique_id):
        B = image.shape[0]
        chunks = []
        mask_chunks = []
        total_batches = (B + per_batch - 1) // per_batch
        current_batch = 0

        for start_idx in range(0, B, per_batch):
            current_batch += 1
            end_idx = min(start_idx + per_batch, B)
            sub_img = image[start_idx:end_idx]
            sub_mask = mask[start_idx:end_idx] if mask is not None else None

            sub_out_img, sub_out_mask = process_fn(sub_img, sub_mask)
            chunks.append(sub_out_img.cpu())
            mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)

            cls._log_progress(
                f"<tr><td>Resize</td><td>batch {current_batch}/{total_batches} · images {end_idx}/{B}</td></tr>",
                unique_id,
            )

        out_image = torch.cat(chunks, dim=0)
        if any(m is not None for m in mask_chunks):
            out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
        else:
            out_mask = None
        return out_image, out_mask

    # ------------------------------------------------------------ outpaint

    @classmethod
    def _run_outpaint(cls, image, mask, width, height, divisible_by, pad_color, crop_position,
                       device, per_batch, unique_id):
        """
        OUTPAINT_MODE: the image content is never scaled. It is placed as-is
        onto a new canvas of (width, height), anchored per crop_position.
        Dimensions where the new canvas is smaller than the image are cropped
        instead of padded. The returned mask is always populated (1.0 = new
        canvas area to outpaint, 0.0 = original image area / an existing
        input mask's values there), which is what a downstream inpainting
        sampler expects.
        """
        B, H, W, C = image.shape

        target_width = width if width != 0 else W
        target_height = height if height != 0 else H

        if divisible_by > 1:
            target_width -= target_width % divisible_by
            target_height -= target_height % divisible_by
        target_width = max(target_width, 1)
        target_height = max(target_height, 1)

        pad_left, pad_right, pad_top, pad_bottom = cls._compute_padding(
            target_width, target_height, W, H, crop_position
        )

        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()
                est_mb = (B * target_height * target_width * C * bytes_per_elem) / (1024 * 1024)
                cls._log_progress(
                    f"<tr><td>Resize</td><td>estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}</td></tr>",
                    unique_id,
                )
            except Exception:
                pass

        def _process_subbatch(in_image, in_mask):
            in_image = in_image if in_image.device == device else in_image.to(device)
            in_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))
            return cls._compose_outpaint_canvas(
                in_image, in_mask, target_width, target_height, pad_left, pad_right, pad_top, pad_bottom, pad_color
            )

        if per_batch == 0 or B <= per_batch:
            return _process_subbatch(image, mask)
        return cls._process_in_chunks(image, mask, per_batch, _process_subbatch, unique_id)

    @staticmethod
    def _axis_placement(diff_low, diff_high, target_len, src_len):
        """
        Given how much of a target-vs-source diff falls on the low/high side
        of one axis (from _compute_padding, positive = pad / negative = crop),
        work out where the source lands on the canvas, where it starts being
        read from in the source, and how many pixels overlap.
        """
        canvas_start = max(diff_low, 0)
        src_start = max(-diff_low, 0)
        overlap = target_len - canvas_start - max(diff_high, 0)
        overlap = max(0, min(overlap, src_len - src_start, target_len - canvas_start))
        return canvas_start, src_start, overlap

    @classmethod
    def _compose_outpaint_canvas(cls, image, mask, target_width, target_height,
                                  pad_left, pad_right, pad_top, pad_bottom, pad_color):
        B, H, W, C = image.shape

        x0, sx0, ow = cls._axis_placement(pad_left, pad_right, target_width, W)
        y0, sy0, oh = cls._axis_placement(pad_top, pad_bottom, target_height, H)

        rgb = ImagePad.parse_color(pad_color, C)
        canvas = image.new_zeros((B, target_height, target_width, C))
        for c in range(C):
            canvas[..., c] = rgb[c]

        canvas_mask = image.new_ones((B, target_height, target_width))  # default: 1.0 = outpaint area

        if ow > 0 and oh > 0:
            canvas[:, y0:y0 + oh, x0:x0 + ow, :] = image[:, sy0:sy0 + oh, sx0:sx0 + ow, :]
            region_mask = mask[:, sy0:sy0 + oh, sx0:sx0 + ow] if mask is not None else image.new_zeros((B, oh, ow))
            canvas_mask[:, y0:y0 + oh, x0:x0 + ow] = region_mask

        return canvas, canvas_mask