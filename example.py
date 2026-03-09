# NOTE: MaximusLLM is still getting improved
#       The current version is a proof-of-concept
#       Focus is on the efficiency and benefits of RandNLA Attention and MAXIS Loss

from src.model import Model, Config
from src.lora import blockswap_attention_layers
from src.infer import test_general_talking
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
import argparse
import torch

def main(prompt):
    repo_id = "yousefg/MaximusLLM"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config.from_pretrained(repo_id)
    model = Model(config, device)
    blockswap_attention_layers(model)

    ckpt = load_file(hf_hub_download(repo_id, "model.safetensors"))
    model.load_state_dict(ckpt)
    model.create_lm_head()

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    test_general_talking(model, tokenizer, prompt=prompt, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help='The prompt message to use.')
    args = parser.parse_args()
    main(args.prompt)
