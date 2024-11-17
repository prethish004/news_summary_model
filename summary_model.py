from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset_path = "./bbc-news-summary/BBC News Summary/News Articles"
texts, summaries = [], []

for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        with open(file_path, 'r') as f:
            text = f.read()
            texts.append(text)
            summaries.append("Manually written summary for " + file_name)  

inputs = tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors="pt")
targets = tokenizer(summaries, max_length=128, truncation=True, padding=True, return_tensors="pt")

dataset = torch.utils.data.TensorDataset(inputs["input_ids"], targets["input_ids"])
dataloader = DataLoader(dataset, batch_size=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(1):  
    for batch in dataloader:
        input_ids, target_ids = batch
        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.save_pretrained("./summary_model")
tokenizer.save_pretrained("./summary_model")
