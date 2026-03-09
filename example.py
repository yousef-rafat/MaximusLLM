from src.model import Model, Config
from src.lora import blockswap_attention_layers
from src.infer import test_general_talking
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
import torch

repo_id = "yousefg/MaximusLLM"
device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config.from_pretrained(repo_id)
model = Model(config, device)
blockswap_attention_layers(model)

ckpt = load_file(hf_hub_download(repo_id, "model.safetensors"))
model.load_state_dict(ckpt)
model.create_lm_head()

tokenizer = AutoTokenizer.from_pretrained(repo_id)

prompt = "Prompt Here..."

test_general_talking(model, tokenizer, prompt=prompt, device=device)
