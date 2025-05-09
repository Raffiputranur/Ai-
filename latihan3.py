from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased", num_labels=2)

# Dataset dummy
texts = ["Produk ini sangat bagus" for _ in range(250)] + ["Sangat mengecewakan dan buruk" for _ in range(250)]
labels = [1]*250 + [0]*250

dataset = SentimentDataset(texts, labels, tokenizer)

training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8, logging_dir="./logs", logging_steps=10)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# t-SNE
embeddings = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings[:300])
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
plt.title("t-SNE Word Embeddings")
plt.savefig("tsne_visualization.png")