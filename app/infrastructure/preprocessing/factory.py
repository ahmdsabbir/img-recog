from app.config import settings
from app.interfaces.preprocessor import I_ImagePreprocessor


def make_preprocessor(config=settings) -> I_ImagePreprocessor:
    """
    Instantiate and return the correct preprocessor based on config.

    Reads:
        config.USE_BG_REMOVAL    (bool)  → RembgPreprocessor or Passthrough
        config.PREPROCESS_SIZE   (tuple) → target output size, e.g. (224, 224)
        config.PREPROCESS_PADDING (float) → padding fraction, e.g. 0.1
        config.PREPROCESS_BG_COLOR (tuple) → canvas fill, e.g. (255, 255, 255)

    Returns:
        I_ImagePreprocessor implementation

    Raises:
        EnvironmentError: if USE_BG_REMOVAL=True but rembg is not installed
    """
    # Import here to avoid circular imports and to keep rembg optional
    from app.infrastructure.preprocessing.rembg_preprocessor import RembgPreprocessor
    from app.infrastructure.preprocessing.passthrough_preprocessor import (
        PassthroughPreprocessor,
    )

    # Shared kwargs that both preprocessors accept
    shared_kwargs = {
        "target_size": getattr(config, "PREPROCESS_SIZE", (224, 224)),
        "bg_color": getattr(config, "PREPROCESS_BG_COLOR", (255, 255, 255)),
    }

    if getattr(config, "USE_BG_REMOVAL", True):
        return RembgPreprocessor(
            **shared_kwargs,
            padding_fraction=getattr(config, "PREPROCESS_PADDING", 0.1),
        )

    return PassthroughPreprocessor(**shared_kwargs)
