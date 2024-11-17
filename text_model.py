from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import os

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

dataset_path = "./bbc-news-summary/BBC News Summary/News Articles"
texts, labels = [], []
label_map = {class_name: idx for idx, class_name in enumerate(os.listdir(dataset_path))}
for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        with open(file_path, 'r') as f:
            text = f.read()
            texts.append(text)
            labels.append(label_map[class_dir])

dataset = TextDataset(texts, labels, tokenizer, max_len=512)
dataloader = DataLoader(dataset, batch_size=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(1):  
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.save_pretrained("./categorization_model")
tokenizer.save_pretrained("./categorization_model")
