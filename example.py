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

def main(args):
    repo_id = "yousefg/MaximusLLM"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config.from_pretrained(repo_id)
    model = Model(config, device)
    blockswap_attention_layers(model)

    ckpt = load_file(hf_hub_download(repo_id, "model.safetensors"))
    model.load_state_dict(ckpt)
    model.create_lm_head()

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    test_general_talking(model, tokenizer, args=args, prompt=args.prompt, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True, type=str, help='The prompt message to use.')
    parser.add_argument('--temperature', type=float, default=0.7, min=0, help='The temperature for the logits')
    parser.add_argument('--ignore_persona', type=bool, action='store_true', default=False, help='Whether to try to model with user/model personas (e.g. <start_of_turn>user)')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--repetition_penalty', type=float, min = 0, default=1.1)
    parser.add_argument('--max_new_tokens', type=int, min = 0, default=100)
    args = parser.parse_args()
    main(args)
