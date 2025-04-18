import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os

#from transformers.optimization import AdamW


# Paths relative to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATASET_PATH = os.path.join(ROOT_DIR, "dataset", "training_data.csv")
BAD_WORDS_PATH = os.path.join(ROOT_DIR, "extradata", "bad_words.csv")

MODEL_DIR = os.path.join(ROOT_DIR, "model")

# Load datasets
df = pd.read_csv(DATASET_PATH)
bad_words_df = pd.read_csv(BAD_WORDS_PATH)
bad_words = set(bad_words_df["word"].str.lower())

# Multi-label binarization
labels = ["EAR", "EYE", "HRT", "GYN", "WTC"]
mlb = MultiLabelBinarizer(classes=labels)
label_matrix = mlb.fit_transform(df["specialties"].str.split(","))

# Preprocess notes: Remove bad words
def preprocess_text(text, bad_words):
    tokens = text.lower().split()
    filtered_tokens = [token for token in tokens if token not in bad_words]
    return " ".join(filtered_tokens)

df["processed_notes"] = df["note"].apply(lambda x: preprocess_text(x, bad_words))

# Dataset class
class SymptomDataset(Dataset):
    def __init__(self, notes, labels, tokenizer, max_len=128):
        self.notes = notes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.notes)
    
    def __getitem__(self, idx):
        note = self.notes[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(note, max_length=self.max_len, padding="max_length", 
                                 truncation=True, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # Placeholder for DistilBioBERT
#todo use Bio Bert or Distil Bio Bert if available 
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(labels),
    problem_type="multi_label_classification"
)
model.to(device)

# Prepare data
dataset = SymptomDataset(df["processed_notes"].tolist(), label_matrix, tokenizer)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training function
def train_model(model, train_loader, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Train and save
train_model(model, train_loader)
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
np.save(os.path.join(MODEL_DIR, "labels.npy"), labels)

if __name__ == "__main__":
    print("Training completed. Model saved to:", MODEL_DIR)
