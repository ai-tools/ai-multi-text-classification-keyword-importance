# AI Multi Text  Classification with Keyword Importance

This project implements a multi-label ```  classification system to predict medical specialties (e.g., `hrt`, `ear`, `WTC`, `EYE`, `gyn`) from symptom notes, with additional functionality to identify key contributing keywords using **Integrated Gradients** and **Attention Weights**. Built using PyTorch, Hugging Face Transformers, and Captum, it trains a transformer-based model (e.g., DistilBERT) and provides inference capabilities with interpretability.

## Project Purpose
The goal is to:
1. Train a multi-label classification model on symptom notes to predict associated medical specialties.
2. Provide interpretability by identifying the most important keywords influencing predictions using:
   - **Integrated Gradients**: Gradient-based attribution method. (Slow but better than SHAP)
   - **Attention Weights**: Attention mechanism from the transformer model. (Very Fast, Acceptable Accuracy)
3. Support preprocessing to filter out unwanted words (e.g., profanity) from notes.
4. To overcome the slowness and limit of SHAP keyword importance


---

###  Why This Approach Over SHAP?
The added section compares Integrated Gradients (IG) and Attention Weights to SHAP, focusing on:
- **Direct Integration**: IG and Attention Weights leverage the transformer’s internal structure (gradients and attention), while SHAP is model-agnostic and less tailored to transformers.
- **Efficiency**: IG and Attention are faster than SHAP’s sampling-based method, especially for complex models like transformers.
- **Granularity**: IG provides token-level insights, and Attention Weights highlight model focus, both more specific than SHAP’s broader feature-level approach.
- **Domain Fit**: These methods suit NLP and transformers better than SHAP, which is more general-purpose.
- **Implementation**: Easier setup with Captum and native attention outputs compared to SHAP’s external library requirements.

-------

Why Attention is Faster Than Integrated Gradients?
Computation Steps:
Attention Weights: Extracted directly from the transformer model’s forward pass when output_attentions=True is set. This requires no additional computation beyond the standard inference, as the attention weights are a byproduct of the model’s normal operation. For a single input, it’s just one forward pass, making it O(1) in terms of extra passes.
Integrated Gradients: Requires multiple forward and backward passes to approximate the gradient integral along a path from a baseline (e.g., zero embeddings) to the input. Typically, IG uses 20–50 steps (configurable), meaning 20–50 forward-backward passes per input, plus gradient computations, making it O(n) where n is the number of steps.
Gradient Overhead:
Attention: No gradient computation is needed; it uses precomputed attention scores (e.g., softmax outputs from each attention head), which are already part of the model’s inference.
IG: Relies on gradient backpropagation through the entire model for each step, which is computationally expensive, especially for deep transformer models with many layers and parameters.
Memory Usage:
Attention: Only stores the attention weights (e.g., a tensor of shape [num_layers, batch_size, num_heads, seq_len, seq_len]), which is lightweight and processed once.
IG: Requires storing intermediate gradients and embeddings for each step, increasing memory demand, especially for long sequences or large batch sizes.
Implementation Simplicity:
Attention: Natively supported by Hugging Face transformers with minimal post-processing (e.g., averaging across layers/heads), executed in a single torch.no_grad() block.
IG: Uses Captum, necessitating a custom forward function, baseline definition, and gradient tracking (requires_grad=True), adding complexity and runtime overhead.
Practical Impact:
The project (e.g., infer_with_attention vs. infer_with_ig), Attention Weights compute importance almost instantly after inference, while IG takes significantly longer due to the iterative gradient calculations—potentially 10–50x slower depending on sequence length, model size, and hardware.
In summary, Attention Weights are faster because they leverage existing model outputs with no additional passes or gradient computations, whereas IG demands multiple resource-intensive gradient evaluations.



## Project Structure
ai-multi-``` -classification-keyword-importance/
├── dataset/
│   └── training_data.csv         # Input dataset (notes and specialties)
├── extradata/
│   └── bad_words.csv             # List of words to filter out
├── model/                        # Directory for saved model and tokenizer
├── src/
│   ├── inference/
│   │   └── infer.py              # Inference with IG and Attention
│   └── training/
│       └── train.py              # Model training script
├── infer_best_class.py           # Simplified inference for best class and keyword todo
└── README.md                     # This file


### Key Files
- **`train.py`**: Trains a multi-label classification model using a dataset of symptom notes and saves the trained model.
- **`infer.py`**: Performs inference with detailed predictions and keyword importance using Integrated Gradients and Attention Weights.
(todo)
- **`infer_best_class.py`**: Simplified inference to output the best predicted class and most important keyword.

## Prerequisites
- Python 3.8+
- Dependencies:
  `````` bash 
  pip install torch transformers pandas numpy scikit-learn captum
Setup
Clone the Repository:

git clone <>
cd ai-multi-``` -classification-keyword-importance
Install Dependencies:

  `````` bash 

pip install -r requirements.txt

Prepare Data:

Place your dataset (training_data.csv) in the dataset/ folder with columns:
note: Symptom description (e.g., "Right Hip hurts of thigh").
specialties: Comma-separated specialties (e.g., "hrt").
Place bad_words.csv in the extradata/ folder with a word column listing words to filter.
Example training_data.csv:

note,specialties
"Ear pain and hearing loss reported with dizziness","ear,WTC"

Training the Model
Run the training script to train and save the model:

``` ``` bash  

python src/training/train.py

Default Config: Uses distilbert-base-uncased, 3 epochs, batch size 8, learning rate 2e-5.
```
Inference
Full Inference with Keyword Importance
Run inference with both Integrated Gradients and Attention Weights:

``` bash 

python src/inference/infer.py
Example Output:
``` 

Predictions:
ear: 0.4670
EYE: 0.4451
hrt: 0.5420
gyn: 0.5036
WTC: 0.4861

Keyword Importance (Attention Weights):
Class: hrt
Token: hurts           Importance: 1.2345
Token: hip            Importance: 1.0987

Keyword Importance (Integrated Gradients):
Class: hrt
Token: hurts           Importance: 0.1234
Token: hip            Importance: 0.0987

python infer_best_class.py
Example Output:
``` 






IG - Best Class: hrt, Best Keyword: hurts
Attention - Best Class: hrt, Best Keyword: hurts
# Customization

Model: Replace "distilbert-base-uncased" in train.py with a preferred model (e.g., BioBERT).
Hyperparameters: Adjust epochs, batch_size, or lr in train.py.
Threshold: Modify the 0.5 threshold in infer_with_ig or infer_with_attention for positive class detection.
Top Keywords: Change top_k in infer.py to get more/fewer keywords.
Dataset
The training dataset should follow this format:

## CSV 

Columns: note (string), specialties (comma-separated string).
Size: Minimum 6 samples (as in the original example), but 1000+ recommended for better performance.
Example extension to 1000 samples can be hardcoded or generated (see previous discussions).

## Limitations
Model: Uses distilbert-base-uncased by default, which isn’t domain-specific. Consider fine-tuning on a medical corpus (e.g., BioBERT).
Attention: Aggregates attention across all layers/heads; layer-specific analysis might yield different insights.
Data: Performance depends on dataset quality and size.
Future Improvements
Add validation split and evaluation metrics (e.g., F1-score) in train.py.
Support multi-GPU training.
Integrate a medical-specific tokenizer/model (e.g., dmis-lab/biobert-v1.1).
Enhance keyword importance with visualization tools.
License
This project is unlicensed (public domain). Use and modify as needed.

Contact
For issues or contributions, please open a GitHub issue.

Happy classifying! 🚀

---

### Verification and Unit Tests.

check unit test files in inference/infer folder
