from app.training.train_attribute import train_attribute


def run_train(category: str, attribute: str) -> None:
    """
    Train an attribute classifier for the given category/attribute pair.
    """
    train_attribute(category, attribute)