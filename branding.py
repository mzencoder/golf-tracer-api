"""Small reusable overlays for the exported shot video."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


from pathlib import Path


FONT_PATHS = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _font(size: int):
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def _layer(image: Image.Image, x: int, y: int):
    bgra = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
    return x, y, bgra[:, :, :3].astype(np.float32), bgra[:, :, 3:4].astype(np.float32) / 255


class VideoBranding:
    def __init__(self, width: int, height: int, club_name: str | None,
                 include_watermark: bool = True):
        short = min(width, height)
        margin = max(12, round(short * 0.025))
        self.layers = []

        if include_watermark:
            watermark_height = max(52, min(104, round(short * 0.085)))
            watermark_width = round(watermark_height * 2.8)
            watermark = Image.new("RGBA", (watermark_width, watermark_height))
            draw = ImageDraw.Draw(watermark)
            draw.rounded_rectangle((0, 0, watermark_width - 1, watermark_height - 1),
                                   radius=round(watermark_height * 0.22), fill=(255, 255, 255, 210))
            label = "GOLF TRACER"
            font = _font(round(watermark_height * 0.32))
            bounds = draw.textbbox((0, 0), label, font=font)
            draw.text(((watermark_width - (bounds[2] - bounds[0])) // 2,
                       (watermark_height - (bounds[3] - bounds[1])) // 2 - bounds[1]),
                      label, font=font, fill=(23, 70, 42, 255))
            self.layers.append(_layer(watermark, margin, height - margin - watermark_height))

        if club_name:
            badge_height = max(54, min(120, round(short * 0.085)))
            icon_size = round(badge_height * 0.56)
            gap = round(badge_height * 0.18)
            pad = round(badge_height * 0.23)
            max_width = min(round(width * 0.55), width - 2 * margin)
            font_size = round(badge_height * 0.51)
            measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            while True:
                font = _font(font_size)
                bounds = measure.textbbox((0, 0), club_name, font=font)
                text_width = bounds[2] - bounds[0]
                if text_width + icon_size + gap + 2 * pad <= max_width or font_size <= 14:
                    break
                font_size -= 1
            while text_width + icon_size + gap + 2 * pad > max_width and len(club_name) > 1:
                club_name = club_name[:-2].rstrip("… ") + "…"
                bounds = measure.textbbox((0, 0), club_name, font=font)
                text_width = bounds[2] - bounds[0]
            badge_width = min(max_width, text_width + icon_size + gap + 2 * pad)
            badge = Image.new("RGBA", (badge_width, badge_height))
            draw = ImageDraw.Draw(badge)
            draw.rounded_rectangle((0, 0, badge_width - 1, badge_height - 1),
                                   radius=badge_height // 2, fill=(31, 91, 164, 235))
            icon_center = (pad + icon_size // 2, badge_height // 2)
            icon_radius = max(3, icon_size // 4)
            draw.ellipse((icon_center[0] - icon_radius, icon_center[1] - icon_radius,
                          icon_center[0] + icon_radius, icon_center[1] + icon_radius),
                         fill="white")
            text_x = pad + icon_size + gap
            text_y = (badge_height - (bounds[3] - bounds[1])) // 2 - bounds[1]
            draw.text((text_x, text_y), club_name, font=font, fill="white")
            self.layers.append(_layer(badge, width - margin - badge_width, margin))

    def draw(self, frame: np.ndarray) -> None:
        for x, y, pixels, alpha in self.layers:
            region = frame[y:y + pixels.shape[0], x:x + pixels.shape[1]]
            region[:] = (region.astype(np.float32) * (1 - alpha) + pixels * alpha).astype(np.uint8)
