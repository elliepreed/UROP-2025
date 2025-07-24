import pandas as pd

def score_tse(model, fn: str):
    tse_df = pd.read_csv(fn, sep='\t')

    tse_df["sen_prob"] = pd.Series(dtype=object)
    tse_df["wrong_prob"] = pd.Series(dtype=object)

    # Respect the model's max position limit
    max_length = getattr(model.model.config, "n_positions", 1024)

    for idx, row in tse_df.iterrows():
        try:
            sen_prob, wrong_prob = score_pair(model, row.sen, row.wrong_sen, max_length)
            sen_nll = -sen_prob.sum().item()
            wrong_nll = -wrong_prob.sum().item()

            tse_df.at[idx, "sen_prob"] = sen_prob.tolist()
            tse_df.at[idx, "wrong_prob"] = wrong_prob.tolist()
            tse_df.at[idx, "sen_nll"] = sen_nll
            tse_df.at[idx, "wrong_nll"] = wrong_nll
            tse_df.at[idx, "delta"] = wrong_nll - sen_nll

        except Exception as e:
            print(f"⚠️ Skipping idx {idx} due to error: {e}")
            tse_df.at[idx, ["sen_prob", "wrong_prob", "sen_nll", "wrong_nll", "delta"]] = [None]*5

    return tse_df

def score_pair(ilm_model, sen, wrong_sen, max_length):
    tok = ilm_model.tokenizer

    def truncate(text):
        tokens = tok.encode(text, truncation=True, max_length=max_length)
        return tok.decode(tokens, skip_special_tokens=True)

    sen = truncate(sen)
    wrong_sen = truncate(wrong_sen)

    return ilm_model.sequence_score([sen, wrong_sen], reduction=lambda x: x)
