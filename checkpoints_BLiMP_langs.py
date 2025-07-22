!pip install huggingface_hub pandas torch matplotlib numpy tqdm minicons --quiet

import os
import sys
import pandas as pd
import torch
from glob import glob
from huggingface_hub import login, hf_hub_download, HfApi
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import subprocess

# Login to HF (paste your token when prompted)
login()

# Clone the repository and cd into it
!git clone https://github.com/suchirsalhan/multiblimp.git
os.chdir("multiblimp")

# Add paths to sys.path if needed (for imports inside eval_model.py)
sys.path.append(os.path.abspath("scripts/lm_eval"))
sys.path.append(os.path.abspath("src/lm_eval"))

# Define the list of languages directly
langs = ["eng", "nld", "spa", "ell", "pol"]

# Download data
for l in langs:
    lang_path = hf_hub_download(
        repo_id="jumelet/multiblimp",
        filename=f"{l}/data.tsv",
        repo_type="dataset",
        local_dir='hf_cache/'
    )

# Set the MKL threading layer to GNU to avoid conflicts
os.environ['MKL_THREADING_LAYER'] = 'GNU'

checkpoints = [0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040,
               64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140,
               64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600,
               64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000,
               90000, 100000, 110000, 120000, 128000]

api = HfApi()
all_models = api.list_models(author="catherinearnett")
all_bgts = [model.modelId for model in all_models]

# create results dataframe
results = pd.DataFrame(columns=['model', 'checkpoint', 'l1', 'l2', 'biling_cond', 'l1_acc', 'l2_acc'])
results.to_csv('bgpt_multiblimp_results.csv', mode='w', index=False)

# language code mapping
language_map = {
    'en': 'eng',
    'nl': 'nld',
    'es': 'spa',
    'el': 'ell',
    'pl': 'pol'
}

for m in tqdm(all_bgts):
    if 'B-GPT' in m:
        model_cond = m.replace('catherinearnett/B-GPT_', '')
        parts = m.split('_')
        l1 = parts[1]
        l2 = parts[2]
        cond = parts[3]
        l1_iso = language_map[l1]
        l2_iso = language_map[l2]

        for c in checkpoints:
            m_str = str(m)
            c_str = str(c)
            l1_iso_str = str(l1_iso)
            l2_iso_str = str(l2_iso)
            cond_str = str(cond)

            try:
                # Run evaluation for L1
                result_l1 = subprocess.run([
                    "python", "scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--revision", c_str,
                    "--data_dir", f"hf_cache/{l1_iso_str}/",
                    "--src_dir", "multiblimp",
                    "--results_dir", f"bgpt_multiblimp_results/{model_cond}_{c_str}-{l1_iso_str}",
                    "--cache_dir", "hf_cache/"
                ], check=True, env={**os.environ}, capture_output=True, text=True)

                print(f"Output for L1 evaluation (model {m_str}, checkpoint {c_str}):\n{result_l1.stdout}")

                # Run evaluation for L2
                result_l2 = subprocess.run([
                    "python", "scripts/lm_eval/eval_model.py",
                    "--model_name", m_str,
                    "--revision", c_str,
                    "--data_dir", f"hf_cache/{l2_iso_str}/",
                    "--src_dir", "multiblimp",
                    "--results_dir", f"bgpt_multiblimp_results/{c_str}-{l2_iso_str}",
                    "--cache_dir", "hf_cache/"
                ], check=True, env={**os.environ}, capture_output=True, text=True)

                print(f"Output for L2 evaluation (model {m_str}, checkpoint {c_str}):\n{result_l2.stdout}")

                # Collect results for L1
                l1_results_path = f"bgpt_multiblimp_results/{model_cond}_{c_str}-{l1_iso_str}/hf_cache_{l1_iso_str}_data.tsv"
                df_l1 = pd.read_csv(l1_results_path, sep='\t')
                total_samples_l1 = len(df_l1)
                correct_predictions_l1 = len(df_l1[df_l1['delta'] > 0])
                l1_accuracy = correct_predictions_l1 / total_samples_l1

                # Collect results for L2
                l2_results_path = f"bgpt_multiblimp_results/{c_str}-{l2_iso_str}/hf_cache_{l2_iso_str}_data.tsv"
                df_l2 = pd.read_csv(l2_results_path, sep='\t')
                total_samples_l2 = len(df_l2)
                correct_predictions_l2 = len(df_l2[df_l2['delta'] > 0])
                l2_accuracy = correct_predictions_l2 / total_samples_l2

                # Append results
                new_line = pd.DataFrame({
                    'model': [m_str],
                    'checkpoint': [c],
                    'l1': [l1_iso_str],
                    'l2': [l2_iso_str],
                    'biling_cond': [cond_str],
                    'l1_acc': [l1_accuracy],
                    'l2_acc': [l2_accuracy]
                })
                new_line.to_csv('bgpt_multiblimp_results.csv', mode='a', header=False, index=False)
                print(new_line)

            except subprocess.CalledProcessError as e:
                print(f"Error processing model {m_str} at checkpoint {c_str}:")
                print(f"Command failed with exit code {e.returncode}")
                print("stdout:", e.stdout)
                print("stderr:", e.stderr)
                continue

            except Exception as e:
                print(f"Unexpected error with model {m_str} at checkpoint {c_str}: {e}")
                continue
