from .base_attribute_classifier import BaseAttributeClassifier


class BagAttributeClassifier(BaseAttributeClassifier):

    def get_schema(self):
        return {
            "color": [
                "a red bag",
                "a black bag",
                "a brown bag",
                "a white bag"
            ],
            "material": [
                "a leather bag",
                "a canvas bag",
                "a synthetic bag"
            ],
            "size": [
                "a small bag",
                "a medium bag",
                "a large bag"
            ]
        }
