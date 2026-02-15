"""
app/infrastructure/preprocessing/passthrough_preprocessor.py
--------------------------------------------------------------
No-op preprocessor — skips background removal entirely.

Use this when:
  - dataset is already clean (white-background studio shots)
  - running unit tests and don't want rembg as a dependency
  - want to benchmark CLIP quality without preprocessing
  - in a resource-constrained environment (no GPU / low RAM)

Still normalises images to a consistent square size so CLIP gets
valid input, and still applies square-pad to preserve aspect ratio.
"""

from PIL import Image

from app.interfaces.preprocessor import I_ImagePreprocessor


class PassthroughPreprocessor(I_ImagePreprocessor):
    """
    Lightweight preprocessor that skips background removal.

    Still performs:
        1. RGB conversion      (ensures consistent colour mode)
        2. Square pad          (preserves aspect ratio before resize)
        3. Resize              (to CLIP input size)

    Args:
        target_size: (width, height) expected by the embedding model.
                     Must match CLIP's training size (default 224×224).
        bg_color:    RGB fill colour for the square padding canvas.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        bg_color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.target_size = target_size
        self.bg_color = bg_color

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Minimal pipeline: convert → square pad → resize.

        Args:
            image: Raw PIL Image (any size, any mode)

        Returns:
            RGB PIL Image at self.target_size
        """
        image = image.convert("RGB")
        image = self._pad_to_square(image)
        image = image.resize(self.target_size, Image.LANCZOS)
        return image

    def _pad_to_square(self, image: Image.Image) -> Image.Image:
        """
        Pad image to a square canvas without stretching.
        Identical logic to RembgPreprocessor._pad_to_square
        but operates on RGB directly (no alpha channel needed).
        """
        w, h = image.size
        size = max(w, h)

        canvas = Image.new("RGB", (size, size), self.bg_color)

        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        canvas.paste(image, (offset_x, offset_y))

        return canvas
