from app.services.category_classifier_service import CategoryClassifierService
from app.services.product_attribute_service import ProductAttributeService
from app.services.zero_shot_attribute_service import ZeroShotAttributeService
from cli.message import Message


Msg = Message()

def run_classify(embedding, cache, img_path: str, use_trained: bool = False) -> None:
    """
    Classify the category and attributes of the product image at `img_path`.

    When `use_trained` is True, attempts to use fine-tuned attribute models and
    falls back to zero-shot if they are unavailable.
    """
    category_service = CategoryClassifierService(embedding_model=embedding)
    category, cat_conf = category_service.classify(img_path)
    print(f"Category: {category} (confidence {cat_conf:.2f})")

    attribute_service = _build_attribute_service(
        embedding, cache, category, use_trained
    )
    attributes = attribute_service.classify(img_path, category=category)

    print(Msg.info("\nAttributes:"))
    for attr_name, info in attributes.items():
        print(f" - {attr_name}: {info['value']} (confidence {info['confidence']:.2f})")


def _build_attribute_service(embedding, cache, category: str, use_trained: bool):
    """Return the appropriate attribute service, with fallback logic."""
    if use_trained:
        print(Msg.highlight("\nUsing trained models for attribute classification..."))
        try:
            return ProductAttributeService(embedding_model=embedding, cache=cache)
        except Exception as e:
            print(Msg.alert(f"Error loading trained models: {e}"))
            print(Msg.alert("Falling back to zero-shot classification..."))

    print(Msg.highlight("\nUsing zero-shot classification for attributes..."))
    return ZeroShotAttributeService(embedding_model=embedding)
