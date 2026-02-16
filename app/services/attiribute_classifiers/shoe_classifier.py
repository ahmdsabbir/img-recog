from .base_attribute_classifier import BaseAttributeClassifier


class ShoeAttributeClassifier(BaseAttributeClassifier):
    def get_schema(self):
        return {
            "color": [
                "a red shoe",
                "a black shoe",
                "a white shoe",
                "a blue shoe"
            ],
            "gender": [
                "a men's shoe",
                "a women's shoe",
                "a unisex shoe"
            ],
            "age_group": [
                "a shoe for adults",
                "a shoe for kids"
            ]
        }