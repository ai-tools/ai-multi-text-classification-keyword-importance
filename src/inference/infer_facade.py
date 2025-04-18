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

# Load trained model and bad words with attn_implementation="eager"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, attn_implementation="eager")
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

# Custom forward function for Integrated Gradients
def forward_with_embeddings(embeddings, attention_mask):
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    return outputs.logits

def infer_best_class_and_keyword_ig(note):
    """
    Infers the best class (specialty) and best keyword using Integrated Gradients.
    Returns a tuple: (best_class, best_keyword)
    """
    processed_note = preprocess_text(note, bad_words)
    
    # Tokenize input
    inputs = tokenizer(processed_note, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get embeddings
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)
    
    # Forward pass for predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Get predictions
    predictions = {label: prob for label, prob in zip(labels, probs)}
    best_class = max(predictions.items(), key=lambda x: x[1])[0]
    best_prob = predictions[best_class]
    
    # Threshold check
    if best_prob <= 0.5:
        return None, None
    
    # Integrated Gradients
    ig = IntegratedGradients(forward_with_embeddings)
    baseline = torch.zeros_like(embeddings)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Get class index for best class
    class_idx = labels.index(best_class)
    attributions = ig.attribute(embeddings, baselines=baseline, target=class_idx, 
                               additional_forward_args=(attention_mask,))
    scores = attributions.sum(dim=-1).squeeze().abs().detach().cpu().numpy()
    
    # Filter and sort tokens by importance
    token_scores = [
        (token, score) for token, score in zip(tokens, scores)
        if token.lower() not in bad_words and token not in ["[CLS]", "[SEP]", "[PAD]"]
    ]
    if token_scores:
        best_keyword = sorted(token_scores, key=lambda x: x[1], reverse=True)[0][0]
    else:
        best_keyword = None
    
    return best_class, best_keyword

def infer_best_class_and_keyword_attention(note):
    """
    Infers the best class (specialty) and best keyword using Attention weights.
    Returns a tuple: (best_class, best_keyword)
    """
    processed_note = preprocess_text(note, bad_words)
    
    # Tokenize input
    inputs = tokenizer(processed_note, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Get predictions
    predictions = {label: prob for label, prob in zip(labels, probs)}
    best_class = max(predictions.items(), key=lambda x: x[1])[0]
    best_prob = predictions[best_class]
    
    # Threshold check
    if best_prob <= 0.5:
        return None, None
    
    # Aggregate attention weights
    avg_attention = torch.stack(attentions).mean(dim=0).mean(dim=1).squeeze(0)
    token_attention_scores = avg_attention.sum(dim=0).cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Filter and sort tokens by importance
    token_scores = [
        (token, score) for token, score in zip(tokens, token_attention_scores)
        if token.lower() not in bad_words and token not in ["[CLS]", "[SEP]", "[PAD]"]
    ]
    if token_scores:
        best_keyword = sorted(token_scores, key=lambda x: x[1], reverse=True)[0][0]
    else:
        best_keyword = None
    
    return best_class, best_keyword

if __name__ == "__main__":
    note = "Right Hip hurts of thigh"
    class_ig, keyword_ig = infer_best_class_and_keyword_ig(note)
    class_att, keyword_att = infer_best_class_and_keyword_attention(note)
    print(f"IG - Best Class: {class_ig}, Best Keyword: {keyword_ig}")
    print(f"Attention - Best Class: {class_att}, Best Keyword: {keyword_att}")