from abc import ABC, abstractmethod
from PIL import Image


class I_ImagePreprocessor(ABC):
    """
    Interface for image preprocessors.

    Implementations must accept a raw PIL Image of any size/mode
    and return a preprocessed PIL Image ready for the embedding model.

    The contract guarantees:
        - Input:  any PIL Image (any mode, any size)
        - Output: RGB PIL Image at the embedding model's expected input size
    """

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess a single image.

        Args:
            image: Raw PIL Image (product photo, any size/mode)

        Returns:
            Preprocessed RGB PIL Image ready for CLIP (or other embedder)
        """
        ...

    def preprocess_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """
        Preprocess a list of images.

        Default implementation calls preprocess() in a loop.
        Subclasses may override for batched optimisations (e.g. GPU batching).

        Args:
            images: List of raw PIL Images

        Returns:
            List of preprocessed RGB PIL Images (same order, same length)
        """
        return [self.preprocess(img) for img in images]
