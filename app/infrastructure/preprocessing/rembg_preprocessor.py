"""
Pipeline:
    Raw Image
        → Background Removal   (rembg / U2Net)
        → Tight Crop           (bounding box of foreground pixels)
        → Square Pad           (center product on white/custom canvas)
        → Resize               (to CLIP input size, default 224×224)

Install dependency:
    pip install rembg

The U2Net model weights (~170MB) are downloaded automatically on first run
and cached in ~/.u2net/
"""

import io
import numpy as np
from PIL import Image

from app.interfaces.preprocessor import I_ImagePreprocessor

try:
    from rembg import remove as rembg_remove

    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


class RembgPreprocessor(I_ImagePreprocessor):
    """
    Production preprocessor.

    Removes background with rembg, crops tightly to the product,
    pads to a square canvas, and resizes to the target dimensions.

    Args:
        target_size:      (width, height) expected by the embedding model.
                          Must match what CLIP was trained on (default 224×224).
        padding_fraction: Whitespace to add around the cropped product,
                          as a fraction of its bounding box size.
                          0.1 = 10% padding on each side (recommended).
        bg_color:         RGB tuple for the background canvas fill.
                          (255, 255, 255) = white (matches most studio datasets).

    Raises:
        EnvironmentError: if rembg is not installed and bg_remove is attempted.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        padding_fraction: float = 0.1,
        bg_color: tuple[int, int, int] = (255, 255, 255),
    ):
        if not REMBG_AVAILABLE:
            raise EnvironmentError(
                "rembg is not installed. "
                "Install it with: pip install rembg\n"
                "Or use PassthroughPreprocessor to skip background removal."
            )

        self.target_size = target_size
        self.padding_fraction = padding_fraction
        self.bg_color = bg_color

    # ------------------------------------------
    # Public API (implements IImagePreprocessor)
    # ------------------------------------------
    def preprocess(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGBA")
        img_rgba = self._remove_bg(img_rgba=img)
        img_rgba = self._crop_to_foreground(img_rgba=img_rgba)
        img_rgba = self._pad_to_square(img_rgba=img_rgba)
        img_rgba = self._resize(image=img_rgba)

        return img_rgba

    def _remove_bg(self, img_rgba: Image.Image) -> Image.Image:
        """
        Remove background using rembg (U2Net).

        Converts PIL → PNG bytes → rembg → PIL RGBA.
        The alpha channel marks foreground (255) vs background (0).
        """
        buf = io.BytesIO()
        img_rgba.save(buf, format="PNG")
        result_bytes = rembg_remove(buf.getvalue())

        return Image.open(io.BytesIO(result_bytes)).convert("RGBA")

    def _crop_to_foreground(self, img_rgba: Image.Image) -> Image.Image:
        """
        Crop tightly to the bounding box of non-transparent pixels,
        then add a small padding margin.

        Uses the alpha channel to detect foreground pixels.
        Pixels with alpha > 10 are considered foreground
        (threshold of 10 avoids noise from semi-transparent edges).

        Returns RGBA image of the product + padding.
        """
        alpha = np.array(img_rgba.getchannel("A"))

        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)

        # Edge case: fully transparent image (eg: bad bg removal)
        if not rows.any():
            return img_rgba

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        top, bottom = int(row_indices[0]), int(row_indices[-1])
        left, right = int(col_indices[0]), int(col_indices[-1])

        h = bottom - top
        w = right - left

        pad_y = int(h * self.padding_fraction)
        pad_x = int(w * self.padding_fraction)

        img_h, img_w = alpha.shape

        top = max(0, top - pad_y)
        bottom = min(img_h, bottom + pad_y)
        left = max(0, left - pad_x)
        right = min(img_w, right + pad_x)

        return img_rgba.crop((left, top, right, bottom))

    def _pad_to_square(self, img_rgba: Image.Image) -> Image.Image:
        """
        Paste the product onto a square canvas, centered.

        This preserves aspect ratio before the final resize —
        a tall shoe and a wide handbag both end up correctly shaped
        rather than squashed. The canvas is filled with self.bg_color.

        Returns an RGB image (alpha channel dropped, bg filled in).
        """
        w, h = img_rgba.size
        size = max(w, h)

        canvas = Image.new("RGBA", (size, size), (*self.bg_color, 255))

        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        canvas.paste(img_rgba, (offset_x, offset_y), mask=img_rgba)

        return canvas.convert("RGB")

    def _resize(self, image: Image.Image) -> Image.Image:
        """
        Resize to target_size using LANCZOS (best quality for downsampling).
        """
        return image.resize(self.target_size, Image.LANCZOS)
