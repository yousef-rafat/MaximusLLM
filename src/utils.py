import torch
from tqdm import tqdm
import tempfile
import os
import json
from transformers import AutoTokenizer

# decrease the vocab size of the model
def prune_vocab_safely(tokenizer, dataset_iterator, num_samples_to_scan=100_000):
    
    original_vocab_size = len(tokenizer)
    keep_mask = torch.zeros(original_vocab_size, dtype=torch.bool)
    
    for token_id in list(tokenizer.all_special_ids):
        keep_mask[token_id] = True

    seen_tokens = set()
    
    for i, batch in tqdm(enumerate(dataset_iterator), total=num_samples_to_scan):
        if i >= num_samples_to_scan: break
        
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]

        if isinstance(ids[0], list):
             for seq in ids:
                 seen_tokens.update(seq)
        else:
             seen_tokens.update(ids)

    for tid in seen_tokens:
        if tid < original_vocab_size:
            keep_mask[tid] = True

    keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]
    new_vocab_size = len(keep_indices)
    
    print(f"Original Vocab: {original_vocab_size}")
    print(f"Pruned Vocab:   {new_vocab_size}")
    print(f"Reduction:      {100 * (1 - new_vocab_size/original_vocab_size):.2f}% removed")

    old_to_new_map = torch.full((original_vocab_size,), -1, dtype=torch.long)
    old_to_new_map[keep_indices] = torch.arange(new_vocab_size)
    old_to_new_map = old_to_new_map.tolist()
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer.save_pretrained(temp_dir)
        
        json_path = os.path.join(temp_dir, "tokenizer.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        model_vocab = data['model']['vocab']
        new_vocab_dict = {}

        for token, old_id in model_vocab.items():
            if old_id < original_vocab_size:
                new_id = old_to_new_map[old_id]
                if new_id != -1:
                    new_vocab_dict[token] = new_id
        
        data['model']['vocab'] = new_vocab_dict
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            
        new_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    return keep_indices, new_vocab_size, new_tokenizer

def apply_pruning_to_model(model, keep_indices, new_vocab_size):
    
    emb_layer = model.embed_tokens
        
    new_emb_weight = emb_layer.weight.data[keep_indices].clone()
    emb_layer.weight = torch.nn.Parameter(new_emb_weight)
    emb_layer.num_embeddings = new_vocab_size
    
    new_head_weight = model.lm_head.weight.data[keep_indices].clone()
    model.lm_head.weight = torch.nn.Parameter(new_head_weight)
    model.lm_head.out_features = new_vocab_size
        
    model.config.vocab_size = new_vocab_size
    print("model pruned successfully")
    return model

def save_model_on_hf(model, repo_name = "yousefg/MaximusLLM", token = "", tokenizer=None, config=None):
    # huggingface_hub==0.34.0
    from huggingface_hub import HfApi, HfFolder, create_repo, upload_file, upload_folder

    HfFolder.save_token(token)
    api = HfApi()

    create_repo(repo_name, repo_type="model", private=False, exist_ok=True, token=token)

    local_model_path = "/src/model.pt"

    upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo="/src/model.pt",
        repo_id=repo_name,
        repo_type="model",
        token=token,
    )

    if tokenizer is not None:
        with tempfile.TemporaryDirectory() as temp:
            tokenizer.save_pretrained(temp)
            upload_folder(
                folder_path=temp,
                path_in_repo="tokenizer",
                repo_id=repo_name,
                repo_type="model",
                token=token,
            )
