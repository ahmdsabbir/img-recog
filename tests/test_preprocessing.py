import pytest
from PIL import Image

from app.infrastructure.preprocessing.passthrough_preprocessor import (
    PassthroughPreprocessor,
)

# Conditionally import rembg preprocessor
try:
    from app.infrastructure.preprocessing.rembg_preprocessor import RembgPreprocessor

    REMBG_AVAILABLE = True
except (ImportError, EnvironmentError, SystemExit, Exception):
    REMBG_AVAILABLE = False


def make_image(
    width: int, height: int, color=(200, 100, 50), mode="RGB"
) -> Image.Image:
    """Create a solid-color test image."""
    if mode == "L":
        # Grayscale needs a single int, not an RGB tuple
        # Convert RGB tuple to luminance value
        single = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        return Image.new(mode, (width, height), single)
    return Image.new(mode, (width, height), color)


def make_rgba_image(width: int, height: int) -> Image.Image:
    """Create an RGBA test image with a centered white square on transparent bg."""

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Draw a white square in the center (simulates a product)
    cx, cy = width // 2, height // 2
    size = min(width, height) // 3

    for x in range(cx - size, cx + size):
        for y in range(cy - size, cy + size):
            img.putpixel((x, y), (255, 255, 255, 255))

    return img


# --------------------------------------------
# PassthroughPreprocessor tests
# --------------------------------------------


class TestPassthroughPreprocessor:
    def test_output_size_default(self):
        """Output must be 224×224 by default."""
        pp = PassthroughPreprocessor()
        result = pp.preprocess(make_image(640, 480))
        assert result.size == (224, 224)

    def test_output_size_custom(self):
        """Custom target_size is respected."""
        pp = PassthroughPreprocessor(target_size=(128, 128))
        result = pp.preprocess(make_image(300, 400))
        assert result.size == (128, 128)

    def test_output_mode_is_rgb(self):
        """Output must always be RGB regardless of input mode."""
        pp = PassthroughPreprocessor()
        for mode in ["L", "RGBA", "RGB", "P"]:
            img = (
                make_image(200, 200, mode=mode)
                if mode != "P"
                else make_image(200, 200).convert("P")
            )
            result = pp.preprocess(img)
            assert result.mode == "RGB", f"Expected RGB for input mode {mode}"

    def test_square_image_unchanged_proportions(self):
        """A square image should not be distorted."""
        pp = PassthroughPreprocessor()
        result = pp.preprocess(make_image(300, 300))
        assert result.size == (224, 224)

    def test_portrait_image_no_distortion(self):
        """
        A tall portrait image (e.g. 100×400) padded to a square
        should produce a 224×224 output.
        The product itself should be centered and not stretched.
        """
        pp = PassthroughPreprocessor()
        result = pp.preprocess(make_image(100, 400, color=(255, 0, 0)))
        assert result.size == (224, 224)
        assert result.mode == "RGB"

    def test_landscape_image_no_distortion(self):
        """A wide landscape image should pad and resize correctly."""
        pp = PassthroughPreprocessor()
        result = pp.preprocess(make_image(400, 100, color=(0, 255, 0)))
        assert result.size == (224, 224)
        assert result.mode == "RGB"

    def test_very_small_image(self):
        """A tiny image (e.g. 10×10) should upscale cleanly."""
        pp = PassthroughPreprocessor()
        result = pp.preprocess(make_image(10, 10))
        assert result.size == (224, 224)

    def test_batch_preprocess_length(self):
        """preprocess_batch should return same number of images as input."""
        pp = PassthroughPreprocessor()
        images = [make_image(100, 200) for _ in range(5)]
        results = pp.preprocess_batch(images)
        assert len(results) == 5

    def test_batch_preprocess_all_correct_size(self):
        """Every image in a batch should be 224×224."""
        pp = PassthroughPreprocessor()
        images = [make_image(w, h) for w, h in [(100, 200), (300, 300), (50, 400)]]
        results = pp.preprocess_batch(images)
        for r in results:
            assert r.size == (224, 224)

    def test_bg_color_applied(self):
        """
        A portrait image padded to square should have the bg_color
        in the left/right padding strips.
        """
        bg = (255, 0, 0)  # red background
        pp = PassthroughPreprocessor(target_size=(100, 100), bg_color=bg)
        # Tall image → left/right strips will be bg_color
        img = make_image(50, 200, color=(0, 0, 255))  # blue product
        result = pp.preprocess(img)
        # Top-left corner pixel should be the bg color (it's in the padding strip)
        assert result.size == (100, 100)


# --------------------------------------------
# RembgPreprocessor tests (skipped if not installed)
# --------------------------------------------


@pytest.mark.skipif(not REMBG_AVAILABLE, reason="rembg not installed")
class TestRembgPreprocessor:
    def test_output_size(self):
        """Output must be 224×224."""
        pp = RembgPreprocessor()
        result = pp.preprocess(make_image(640, 480))
        assert result.size == (224, 224)

    def test_output_mode_is_rgb(self):
        """Output must be RGB."""
        pp = RembgPreprocessor()
        result = pp.preprocess(make_image(300, 300))
        assert result.mode == "RGB"

    def test_custom_target_size(self):
        """Custom target_size is respected."""
        pp = RembgPreprocessor(target_size=(128, 128))
        result = pp.preprocess(make_image(300, 300))
        assert result.size == (128, 128)

    def test_rgba_input_handled(self):
        """RGBA input (e.g. PNG with transparency) is handled."""
        pp = RembgPreprocessor()
        img = make_rgba_image(300, 300)
        result = pp.preprocess(img.convert("RGB"))
        assert result.size == (224, 224)
        assert result.mode == "RGB"

    def test_batch_length(self):
        """Batch output has same length as input."""
        pp = RembgPreprocessor()
        imgs = [make_image(100, 150) for _ in range(3)]
        results = pp.preprocess_batch(imgs)
        assert len(results) == 3


# --------------------------------------------
# Interface contract test (both must pass)
# --------------------------------------------


@pytest.mark.parametrize(
    "PreprocessorClass,kwargs",
    [
        (PassthroughPreprocessor, {}),
        pytest.param(
            "RembgPreprocessor",
            {},
            marks=pytest.mark.skipif(not REMBG_AVAILABLE, reason="rembg not installed"),
        ),
    ],
)
def test_interface_contract(PreprocessorClass, kwargs):
    """
    Both preprocessors must satisfy the IImagePreprocessor contract:
    given any RGB image, return a 224×224 RGB image.
    """
    if PreprocessorClass == "RembgPreprocessor":
        from app.infrastructure.preprocessing.rembg_preprocessor import (
            RembgPreprocessor,
        )

        PreprocessorClass = RembgPreprocessor

    pp = PreprocessorClass(**kwargs)
    img = make_image(320, 240)
    result = pp.preprocess(img)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (224, 224)
