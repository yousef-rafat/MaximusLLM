import torch
from huggingface_hub import upload_file, delete_file
from safetensors.torch import save_file
import tempfile
import os

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

def get_raw_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_model"):
        model = model._orig_model
    return model
