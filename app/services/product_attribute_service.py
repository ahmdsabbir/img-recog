from app.interfaces.embedding import I_EmbeddingModel
from app.services.attiribute_classifiers.shoe_classifier import ShoeAttributeClassifier
from app.services.attiribute_classifiers.bag_classifier import BagAttributeClassifier


class ProductAttributeService:
    CLASSIFIER_REGISTRY = {
        "shoe": ShoeAttributeClassifier,
        "bag": BagAttributeClassifier
    }


    def __init__(self, embedding_model: I_EmbeddingModel):
        self.embedding_model = embedding_model


    def _get_classifier(self, category: str):
        classifier_class = self.CLASSIFIER_REGISTRY.get(category)

        if classifier_class is None:
            raise Exception(f'category: {category} not found')
            
        return classifier_class(self.embedding_model)
    

    def classify(self, img_path: str, category: str):
        classifier = self._get_classifier(category=category)

        if classifier is None:
            return {}
        
        return classifier.classify(img_path=img_path)