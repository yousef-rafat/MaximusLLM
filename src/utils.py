import torch
from huggingface_hub import upload_file, delete_file
from safetensors.torch import save_file
import tempfile
import os
import torch.distributed as dist

def update_model_hf(model_path, hf_dir="yousefg/MaximusLLM", token="", full_replace=False):
    with tempfile.TemporaryDirectory() as temp_dir:
        
        file_to_upload = model_path
        
        if model_path.endswith(".pt"):            
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            for k, v in state_dict.items():
                state_dict[k] = v.to(torch.float16)

            shared_keys = ['module._orig_mod.lm_head.weight', 'module._orig_mod.embed_tokens.weight']

            for key in shared_keys:
                if key in state_dict:
                    state_dict[key] = state_dict[key].clone().contiguous()

            temp_safe_path = os.path.join(temp_dir, "model.safetensors")
            save_file(state_dict, temp_safe_path)
            
            file_to_upload = temp_safe_path

        upload_file(
            path_or_fileobj=file_to_upload,
            path_in_repo="model_test.safetensors" if not full_replace else "model.safetensors",
            repo_id=hf_dir,
            token=token
        )

        if full_replace:
            delete_file(
                path_in_repo="model_test.safetensors",
                repo_id = hf_dir,
                token=token
            )

def clean_checkpoint(checkpoint):
    new_state_dict = {}

    for k, v in checkpoint.items():
        new_k = k.replace("module.", "")
        new_k = new_k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
        
    return new_state_dict

def save_maximus_checkpoint(model, path):
    state_dict = model.state_dict()
    
    clean_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("_orig_mod.", "")
        clean_dict[new_k] = v
        
    save_file(clean_dict, path)
    print(f"checkpoint saved to {path}")

def get_global_loss(running_loss, world_size):
    if not (world_size > 1):
        return running_loss
    t = torch.tensor([running_loss], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size
