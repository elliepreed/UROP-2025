import pandas as pd
import os


def score_tse(model, fn: str):
    tse_df = pd.read_csv(fn, sep='\t')

    tse_df["sen_prob"] = pd.Series(dtype=object).astype(object)
    tse_df["wrong_prob"] = pd.Series(dtype=object).astype(object)

    # Set max_length based on model context window (commonly 1024 or 2048)
    max_length = 1024

    for idx, row in tse_df.iterrows():
        sen_prob, wrong_prob = score_pair(model, row.sen, row.wrong_sen, max_length)

        sen_nll = -sen_prob.sum().item()
        wrong_nll = -wrong_prob.sum().item()

        tse_df.at[idx, "sen_prob"] = sen_prob.tolist()
        tse_df.at[idx, "wrong_prob"] = wrong_prob.tolist()

        tse_df.loc[idx, "sen_nll"] = sen_nll
        tse_df.loc[idx, "wrong_nll"] = wrong_nll
        tse_df.loc[idx, "delta"] = wrong_nll - sen_nll

    return tse_df


def score_pair(ilm_model, sen, wrong_sen, max_length=1024):
    tokenizer = ilm_model.tokenizer

    # Tokenize and truncate to max_length
    sen_tokens = tokenizer.tokenize(sen)[:max_length]
    wrong_tokens = tokenizer.tokenize(wrong_sen)[:max_length]

    # Reconstruct string from tokens
    sen_trunc = tokenizer.convert_tokens_to_string(sen_tokens)
    wrong_trunc = tokenizer.convert_tokens_to_string(wrong_tokens)

    stimuli = [sen_trunc, wrong_trunc]
    return ilm_model.sequence_score(stimuli, reduction=lambda x: x)
