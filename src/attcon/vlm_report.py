from __future__ import annotations

"""Image renderers for the Stage 7 external vision-language audit."""

import base64
import hashlib
import io
import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .nl_report import NLExample


VLM_MODES = {
    "visual_latent_state",
    "visual_observation_only",
    "visual_symbolic_state",
}

LATENT_PALETTE = (
    (68, 1, 84),
    (70, 50, 126),
    (54, 92, 141),
    (39, 127, 142),
    (31, 161, 135),
    (74, 193, 109),
    (160, 218, 57),
    (253, 231, 37),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size=size)
    except OSError:  # pragma: no cover - platform-specific font fallback.
        return ImageFont.load_default()


def _latent_heatmap_png(example: NLExample) -> bytes:
    """Render a fixed-layout, label-free activation heatmap."""

    values = example.controller_state.detach().float().cpu().reshape(-1).numpy()
    columns = 16
    rows = max(math.ceil(values.size / columns), 1)
    cell = 28
    legend_height = cell
    image = Image.new(
        "RGB",
        (columns * cell, rows * cell + legend_height),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(image)
    for index, raw_value in enumerate(values):
        row, column = divmod(index, columns)
        level = min(7, max(0, int(round((float(raw_value) + 1.0) * 3.5))))
        x0 = column * cell
        y0 = row * cell
        draw.rectangle(
            (x0, y0, x0 + cell - 2, y0 + cell - 2),
            fill=LATENT_PALETTE[level],
        )
    legend_y = rows * cell
    legend_width = (columns * cell) // len(LATENT_PALETTE)
    for level, color in enumerate(LATENT_PALETTE):
        x0 = level * legend_width
        x1 = (level + 1) * legend_width - 2
        draw.rectangle((x0, legend_y + 4, x1, legend_y + cell - 1), fill=color)
    return _encode_png(image)


def _text_panel_png(text: str, *, font_size: int) -> bytes:
    lines = text.splitlines() or [""]
    font = _font(font_size)
    line_height = font_size + 5
    width = 900
    height = max(160, 28 + line_height * len(lines))
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    y = 14
    for line in lines:
        draw.text((16, y), line, fill="black", font=font)
        y += line_height
    return _encode_png(image)


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_vlm_png(example: NLExample, mode: str) -> bytes:
    if mode == "visual_latent_state":
        return _latent_heatmap_png(example)
    if mode == "visual_observation_only":
        return _text_panel_png(example.observation_only, font_size=18)
    if mode == "visual_symbolic_state":
        return _text_panel_png(example.symbolic_state, font_size=17)
    raise ValueError(f"unknown VLM mode: {mode}")


class VLMImageRenderer:
    """Cache image inputs and expose builders accepted by ``run_nl_report_mode``."""

    def __init__(self, mode: str):
        if mode not in VLM_MODES:
            raise ValueError(f"unknown VLM mode: {mode}")
        self.mode = mode
        self._cache: dict[int, tuple[str, dict[str, Any]]] = {}

    def _render(self, example: NLExample) -> tuple[str, dict[str, Any]]:
        key = id(example)
        if key not in self._cache:
            png = render_vlm_png(example, self.mode)
            encoded = base64.b64encode(png).decode("ascii")
            self._cache[key] = (
                f"data:image/png;base64,{encoded}",
                {
                    "kind": self.mode,
                    "sha256": hashlib.sha256(png).hexdigest(),
                    "png_bytes": len(png),
                    "contains_symbolic_field_names": self.mode == "visual_symbolic_state",
                    "contains_observation_field_names": self.mode
                    in {"visual_observation_only", "visual_symbolic_state"},
                },
            )
        return self._cache[key]

    def content(self, example: NLExample) -> list[dict[str, Any]]:
        data_url, _ = self._render(example)
        prompt = {
            "visual_latent_state": "Decode the controller report from this opaque state heatmap.",
            "visual_observation_only": "Report what can be inferred from this observation-only panel.",
            "visual_symbolic_state": "Read this explicit symbolic state panel and report it faithfully.",
        }[self.mode]
        return [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_url, "detail": "high"},
        ]

    def summary(self, example: NLExample) -> dict[str, Any]:
        _, summary = self._render(example)
        return summary
