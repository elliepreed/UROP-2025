def score_pair(ilm_model, sen, wrong_sen, max_length=1024):
    """
    Score a correct vs incorrect sentence using an ILM model.
    
    Args:
        ilm_model: An object with `.tokenizer` and `.sequence_score` method.
        sen (str): Correct sentence.
        wrong_sen (str): Incorrect sentence.
        max_length (int): Max number of tokens allowed (default 1024).
    
    Returns:
        A tuple of scores (correct_score, wrong_score).
    """
    tokenizer = ilm_model.tokenizer

    # Tokenize and truncate using token IDs
    sen_ids = tokenizer.encode(sen, truncation=True, max_length=max_length)
    wrong_ids = tokenizer.encode(wrong_sen, truncation=True, max_length=max_length)

    # Convert token IDs back to text (optional, if model expects text)
    sen_trunc = tokenizer.decode(sen_ids, skip_special_tokens=True)
    wrong_trunc = tokenizer.decode(wrong_ids, skip_special_tokens=True)

    # Score both using ILM model
    return ilm_model.sequence_score([sen_trunc, wrong_trunc], reduction=lambda x: x)
