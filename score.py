import pandas as pd
import os

def score_tse(model, fn: str):
    tse_df = pd.read_csv(fn, sep='\t')

    tse_df["sen_prob"] = pd.Series(dtype=object).astype(object)
    tse_df["wrong_prob"] = pd.Series(dtype=object).astype(object)

    # Set max_length based on model config (typically 1024 for GPT-2)
    max_length = getattr(model.model.config, "n_positions", 1024)

    for idx, row in tse_df.iterrows():
        try:
            sen_prob, wrong_prob = score_pair(model, row.sen, row.wrong_sen, max_length)

            sen_nll = -sen_prob.sum().item()
            wrong_nll = -wrong_prob.sum().item()

            tse_df.at[idx, "sen_prob"] = sen_prob.tolist()
            tse_df.at[idx, "wrong_prob"] = wrong_prob.tolist()

            tse_df.loc[idx, "sen_nll"] = sen_nll
            tse_df.loc[idx, "wrong_nll"] = wrong_nll
            tse_df.loc[idx, "delta"] = wrong_nll - sen_nll
        except Exception as e:
            print(f"⚠️ Skipping example at index {idx} due to error: {e}")
            tse_df.loc[idx, ["sen_prob", "wrong_prob", "sen_nll", "wrong_nll", "delta"]] = [None, None, None, None, None]

    return tse_df


def score_pair(ilm_model, sen, wrong_sen, max_length):
    tokenizer = ilm_model.tokenizer

    # Truncate if needed
    def truncate(text):
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens, skip_special_tokens=True)

    sen = truncate(sen)
    wrong_sen = truncate(wrong_sen)

    stimuli = [sen, wrong_sen]

    return ilm_model.sequence_score(stimuli, reduction=lambda x: x)
