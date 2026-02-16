import torch.nn as nn


class AttributeHead(nn.Module):
    """
    Generic classifier head for a single attribute
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)


    def forward(self, x):
        return self.classifier(x)