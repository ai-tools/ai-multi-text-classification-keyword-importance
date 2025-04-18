

def preprocess_text(text, bad_words):
    tokens = text.lower().split()
    filtered_tokens = [token for token in tokens if token not in bad_words]
    return " ".join(filtered_tokens)

