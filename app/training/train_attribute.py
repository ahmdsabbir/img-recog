import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
from app.models.attribute_head import AttributeHead


class DirAttributeDataset(Dataset):
    """
    Dataset that loads images from:

    data/training/<category>/<attribute>/<class_name>/*.jpg
    """

    def __init__(self, base_path: str):
        self.clip_model = ClipEmbeddingModel()
        self.samples = []
        self.class_to_idx = {}

        class_names = sorted(os.listdir(base_path))
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        for cls in class_names:
            cls_path = os.path.join(base_path, cls)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.samples.append((img_path, self.class_to_idx[cls]))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        embedding = self.clip_model.encode_image(img_path)
        embedding = torch.tensor(embedding, dtype=torch.float32)

        return embedding, label


def train_attribute(category: str, attribute: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = f"data/training/{category}/{attribute}"

    dataset = DirAttributeDataset(base_path=base_path)
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    num_classes = len(dataset.class_to_idx)

    model = AttributeHead(embedding_dim=512, num_classes=num_classes).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10

    for epoch in range(epochs):
        total_loss = 0
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # save model
    import json

    model_dir = f"models/{category}/{attribute}"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pt")
    classes_path = os.path.join(model_dir, "classes.json")

    torch.save(model.state_dict(), model_path)

    # Save class mapping as idx->class_name for inference
    idx_to_class = {i: cls for cls, i in dataset.class_to_idx.items()}
    with open(classes_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved classes to {classes_path}")
    print("Class mapping:", dataset.class_to_idx)


if __name__ == '__main__':
    pass