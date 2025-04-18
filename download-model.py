from huggingface_hub import hf_hub_download, snapshot_download

# Download specific files (e.g., tokenizer config)
tokenizer_file = hf_hub_download(
    repo_id="distilbert-base-uncased",
    filename="tokenizer.json"
)
print(f"Tokenizer file downloaded to: {tokenizer_file}")

# Download entire model directory
model_dir = snapshot_download(
    repo_id="distilbert-base-uncased",
    local_dir="./distilbert_model"
)
print(f"Model downloaded to: {model_dir}")

# Load from local files
tokenizer = AutoTokenizer.from_pretrained("./distilbert_model")
model = AutoModelForSequenceClassification.from_pretrained("./distilbert_model")
