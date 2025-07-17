!pip install transformers huggingface_hub pandas torch --quiet

import os
import pandas as pd
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, hf_hub_download

# Login to HF (paste your token when prompted)
login()

# Download MultiBLIMP English and Dutch datasets
eng_path = hf_hub_download(
    repo_id="jumelet/multiblimp",
    filename="eng/data.tsv",
    repo_type="dataset",
    local_dir="hf_cache/",
)
nld_path = hf_hub_download(
    repo_id="jumelet/multiblimp",
    filename="nld/data.tsv",
    repo_type="dataset",
    local_dir="hf_cache/",
)

# Load the datasets
eng_df = pd.read_csv(eng_path, sep="\t")
nld_df = pd.read_csv(nld_path, sep="\t")

# Create folder structure expected by eval code
base_dir = "final_pairs"
phenomenon = "demo_phenomenon"
os.makedirs(f"{base_dir}/{phenomenon}/english", exist_ok=True)
os.makedirs(f"{base_dir}/{phenomenon}/dutch", exist_ok=True)

# Save to .tsv files in that structure
eng_condition = "condition1.tsv"
nld_condition = "condition1.tsv"

eng_df.to_csv(f"{base_dir}/{phenomenon}/english/{eng_condition}", sep="\t", index=False)
nld_df.to_csv(f"{base_dir}/{phenomenon}/dutch/{nld_condition}", sep="\t", index=False)

# Load your B-GPT model and tokenizer
model_name = "catherinearnett/B-GPT_en_nl_simultaneous"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def compute_sentence_nll_batch(sentences, model, tokenizer, device, batch_size=16):
    """Compute NLL for a batch of sentences efficiently"""
    all_nlls = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            # Get per-example loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Calculate NLL for each sentence in batch
            batch_nlls = []
            for j in range(len(batch_sentences)):
                # Get valid tokens (excluding padding)
                valid_length = attention_mask[j].sum().item() - 1  # -1 for shift
                if valid_length > 0:
                    # Calculate cross entropy loss for this sentence
                    sentence_logits = shift_logits[j, :valid_length]
                    sentence_labels = shift_labels[j, :valid_length]

                    loss_fct = torch.nn.CrossEntropyLoss()
                    nll = loss_fct(sentence_logits, sentence_labels).item()
                    batch_nlls.append(nll)
                else:
                    batch_nlls.append(float('inf'))

            all_nlls.extend(batch_nlls)

    return all_nlls

def score_tse_batch(df, model, tokenizer, device):
    """Score all sentences in dataframe at once"""
    # NOTE: Check your TSV columns here!
    # Assuming columns named 'sentence_good' and 'sentence_bad'
    correct_col = "sen"  # Update this to match your column names
    wrong_col = "wrong_sen"     # Update this to match your column names

    # Get all sentences
    correct_sentences = df[correct_col].tolist()
    wrong_sentences = df[wrong_col].tolist()

    # Compute NLLs in batches
    print(f"Computing NLL for {len(correct_sentences)} sentence pairs...")
    correct_nlls = compute_sentence_nll_batch(correct_sentences, model, tokenizer, device)
    wrong_nlls = compute_sentence_nll_batch(wrong_sentences, model, tokenizer, device)

    # Add NLL columns to dataframe
    df = df.copy()
    df['correct_nll'] = correct_nlls
    df['wrong_nll'] = wrong_nlls

    return df

# Evaluate all .tsv files in final_pairs folder
pair_files = glob(f"{base_dir}/**/*.tsv", recursive=True)
print(f"Found {len(pair_files)} files: {pair_files}")
results_dir = os.path.join("model_results", model_name.replace("/", "_"))
os.makedirs(results_dir, exist_ok=True)

for fn in sorted(pair_files):
    print(f"Processing file: {fn}")

    # Read file once
    df = pd.read_csv(fn, sep="\t")

    print(f"Columns: {df.columns.tolist()}")

    # Process entire dataframe at once
    df_with_scores = score_tse_batch(df, model, tokenizer, device)

    # Calculate accuracy
    correct_better = (df_with_scores['correct_nll'] < df_with_scores['wrong_nll']).sum()
    total = len(df_with_scores)
    accuracy = correct_better / total

    print(f"Accuracy: {accuracy:.3f} ({correct_better}/{total})")

    # Save scored dataframe
    phenomenon_lang = fn.split(os.sep)[-3:-1]  # Extract phenomenon and language
    score_fn = os.path.join(results_dir, f"{''.join(phenomenon_lang)}{os.path.basename(fn)}")
    df_with_scores.to_csv(score_fn, sep="\t", index=False)
    print(f"Saved scores to {score_fn}")

print("All files processed!")
