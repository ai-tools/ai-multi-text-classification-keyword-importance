import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import numpy as np
import pandas as pd
import os

# Paths relative to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
BAD_WORDS_PATH = os.path.join(ROOT_DIR, "extradata", "bad_words.csv")

# Load trained model and bad words
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
labels = np.load(os.path.join(MODEL_DIR, "labels.npy"), allow_pickle=True).tolist()
bad_words_df = pd.read_csv(BAD_WORDS_PATH)
bad_words = set(bad_words_df["word"].str.lower())

# Preprocess function
def preprocess_text(text, bad_words):
    tokens = text.lower().split()
    filtered_tokens = [token for token in tokens if token not in bad_words]
    return " ".join(filtered_tokens)

# Custom forward function for IG
def forward_with_embeddings(embeddings, attention_mask):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return outputs.logits

# Inference with IG
def infer_with_ig(note, top_k=5):
    processed_note = preprocess_text(note, bad_words)
    
    # Tokenize input
    inputs = tokenizer(processed_note, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get embeddings from the model's embedding layer
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)  # Enable gradient tracking
    
    # Forward pass for predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Predictions
    predictions = {label: prob for label, prob in zip(labels, probs)}
    print("\nPredictions:")
    for label, prob in predictions.items():
        print(f"{label}: {prob:.4f}")
    
    # Integrated Gradients
    ig = IntegratedGradients(forward_with_embeddings)
    baseline = torch.zeros_like(embeddings)  # Baseline is zero embeddings
    
    # Keyword importance
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    keyword_importance = {}
    for class_idx, label in enumerate(labels):
        if predictions[label] > 0.5:  # Positive predictions only
            attributions = ig.attribute(embeddings, baselines=baseline, target=class_idx, 
                                       additional_forward_args=(attention_mask,))
            # Detach tensor before converting to NumPy
            scores = attributions.sum(dim=-1).squeeze().abs().detach().cpu().numpy()
            token_scores = [(token, score) for token, score in zip(tokens, scores) 
                           if token.lower() not in bad_words]
            top_keywords = sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_k]
            keyword_importance[label] = top_keywords
    
    # Print results
    print("\nKeyword Importance (Integrated Gradients):")
    for label, keywords in keyword_importance.items():
        print(f"\nClass: {label}")
        for token, score in keywords:
            print(f"Token: {token:<15} Importance: {score:.4f}")


# Inference with Attention
def infer_with_attention(note, top_k=5):
    processed_note = preprocess_text(note, bad_words)
    
    # Tokenize input
    inputs = tokenizer(processed_note, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions  # Tuple of attention weights from each layer
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Predictions
    predictions = {label: prob for label, prob in zip(labels, probs)}
    print("\nPredictions:")
    for label, prob in predictions.items():
        print(f"{label}: {prob:.4f}")
    
    # Aggregate attention weights
    # attentions: tuple of (num_layers, batch_size, num_heads, seq_len, seq_len)
    # Average across heads and layers for simplicity
    avg_attention = torch.stack(attentions).mean(dim=0)  # Average across layers
    avg_attention = avg_attention.mean(dim=1)  # Average across heads (batch_size, seq_len, seq_len)
    avg_attention = avg_attention.squeeze(0)  # Remove batch dimension (seq_len, seq_len)
    
    # Sum attention scores for each token (column-wise sum)
    token_attention_scores = avg_attention.sum(dim=0)  # (seq_len,)
    token_attention_scores = token_attention_scores.cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Keyword importance per class
    keyword_importance = {}
    for class_idx, label in enumerate(labels):
        if predictions[label] > 0.5:  # Positive predictions only
            token_scores = [(token, score) for token, score in zip(tokens, token_attention_scores) 
                           if token.lower() not in bad_words and token not in ["[CLS]", "[SEP]", "[PAD]"]]
            top_keywords = sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_k]
            keyword_importance[label] = top_keywords
    
    # Print results
    print("\nKeyword Importance (Attention Weights):")
    for label, keywords in keyword_importance.items():
        print(f"\nClass: {label}")
        for token, score in keywords:
            print(f"Token: {token:<15} Importance: {score:.4f}")

# Test
if __name__ == "__main__":
    note = "Right Hip hurts of thigh"
    infer_with_attention(note)

    infer_with_ig(note)

