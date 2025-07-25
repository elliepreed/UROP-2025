# Install dependencies (skip this if already installed)
!pip install huggingface_hub pandas torch matplotlib numpy tqdm minicons --quiet

# Imports
import os
import sys
import pandas as pd
import torch
import subprocess
from huggingface_hub import login, hf_hub_download, HfApi
from tqdm import tqdm

# Login to Hugging Face
login()

# Clone multiblimp as usual
!git clone https://github.com/elliepreed/multiblimp-1.git
os.chdir("multiblimp-1")

# Use original multiblimp code paths
sys.path.append(os.path.abspath("scripts/lm_eval"))
sys.path.append(os.path.abspath("src/lm_eval"))



# Language codes
langs = [
    "abk", "aqz", "sqi", "amh", "grc", "hbo", "apu", "hye", "eus", "bel", "ben",
    "bho", "bor", "bre", "bul", "bua", "cat", "chu", "xcl", "ces", "dan", "nld",
    "egy", "eng", "myv", "est", "fao", "fin", "fra", "glg", "kat", "deu", "aln",
    "got", "guj", "heb", "azz", "hin", "hit", "hun", "isl", "gle", "ita", "quc",
    "xnr", "krl", "kxh", "kaz", "kir", "koi", "kpv", "lat", "lav", "lij", "lit",
    "olo", "nds", "mkd", "mar", "frm", "ell", "mdf", "yrl", "pcm", "kmr", "sme",
    "fro", "orv", "ota", "fas", "xpg", "pol", "por", "ron", "rus", "san", "gla",
    "hbs", "sms", "slk", "slv", "spa", "arb", "swe", "tam", "ttc", "tpn", "tur",
    "uig", "ukr", "hsb", "urd", "urb", "uzb", "vep", "wbp", "cym", "hyw", "wol",
    "sah", "nhi"
]

# Download data files
for lang in langs:
    print(f"⬇️ Downloading data for: {lang}")
    hf_hub_download(
        repo_id="jumelet/multiblimp",
        filename=f"{lang}/data.tsv",
        repo_type="dataset",
        local_dir='hf_cache/'
    )

# Set MKL flag to avoid threading issues
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Define checkpoints to evaluate
checkpoints = [
    0, 10000, 20000, 50000, 64000, 64050, 64070,  64100, 64120, 
    64160,  64180, 64200,  64500, 64900, 66000,  68000, 80000,
    100000, 120000, 128000
]

# Get all BGPT models
api = HfApi()
all_models = api.list_models(author="catherinearnett")
all_bgts = [model.modelId for model in all_models if 'B-GPT' in model.modelId]

# Prepare CSV for results
results_file = 'bgpt_multiblimp_results.csv'
if not os.path.exists(results_file):
    df = pd.DataFrame(columns=['model', 'checkpoint', 'language', 'accuracy'])
    df.to_csv(results_file, index=False)

# Check if a checkpoint exists on Hugging Face
def checkpoint_exists(model_id, revision):
    try:
        api.model_info(model_id, revision=str(revision))
        return True
    except Exception:
        return False

# Main loop: evaluate all combinations
for m in tqdm(all_bgts):
    for c in checkpoints:
        if not checkpoint_exists(m, c):
            print(f"⚠️ Checkpoint {c} not found for model {m}. Skipping.")
            continue
        for lang in langs:
            try:
                # Run evaluation
                result = subprocess.run([
                    "python", "scripts/lm_eval/eval_model.py",
                    "--model_name", m,
                    "--revision", str(c),
                    "--data_dir", f"hf_cache/{lang}/",
                    "--src_dir", "multiblimp",
                    "--results_dir", f"bgpt_multiblimp_results/{m.replace('/', '_')}_{c}_{lang}",
                    "--cache_dir", "hf_cache/"
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"❌ Eval failed: {m} | checkpoint {c} | language {lang}")
                    print("------ STDOUT ------")
                    print(result.stdout)
                    print("------ STDERR ------")
                    print(result.stderr)
                    continue

                # Load and calculate accuracy
                results_path = f"bgpt_multiblimp_results/{m.replace('/', '_')}_{c}_{lang}/hf_cache_{lang}_data.tsv"
                df_res = pd.read_csv(results_path, sep='\t')
                accuracy = (df_res['delta'] > 0).mean()
                print(f"✅ {m} | ckpt {c} | {lang} → accuracy: {accuracy:.4f}")

                # Save to CSV
                pd.DataFrame({
                    'model': [m],
                    'checkpoint': [c],
                    'language': [lang],
                    'accuracy': [accuracy]
                }).to_csv(results_file, mode='a', header=False, index=False)

            except Exception as e:
                print(f"🔥 Unexpected error: {m} | checkpoint {c} | lang {lang}")
                print(e)
                continue
