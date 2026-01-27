# refine the fisher-svd init to align with the teacher model (distil)
# we only train the head and its norm with an aggressive lr
# made loss from 7.5-8 -> 5.8-6 (300 steps)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import Model, Config
from datasets import load_dataset
from safetensors.torch import save_file, load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from fisher_svd import token

device = "cuda"
torch.set_default_dtype(torch.float32)

teacher = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m", token=token, torch_dtype=torch.float32
).to(device).eval()

config = Config.from_pretrained("yousefg/MaximusLLM")
student = Model(config).to(device)

checkpoint_path = hf_hub_download(repo_id="yousefg/MaximusLLM", filename="model.safetensors", local_dir=".")
clean_ckpt = load_file(checkpoint_path)

if "embed_tokens.weight" in clean_ckpt and "lm_head.weight" not in clean_ckpt:
    clean_ckpt["lm_head.weight"] = clean_ckpt["embed_tokens.weight"]

student.load_state_dict(clean_ckpt, strict=False)

student.requires_grad_(False)

if student.lm_head.weight is student.embed_tokens.weight:
    student.lm_head.weight = nn.Parameter(student.embed_tokens.weight.clone())

student.lm_head.requires_grad_(True)
student.norm.requires_grad_(True)

params = [p for p in student.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(params, lr=1e-3) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
dataset = dataset.skip(300)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", token=token)

student.train()

for step, batch in tqdm(enumerate(dataset), total=300):

    if step >= 300:
        break
    if len(batch['text']) < 100:
        continue
    
    inputs = tokenizer(batch['text'], return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        t_out = teacher(input_ids)
        t_logits = t_out.logits

    s_out = student(input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    s_logits = s_out[0] if isinstance(s_out, tuple) else s_out

    T = 2.0
    loss = F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T**2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 20 == 0:
        print(f"Step {step} | KL Loss: {loss.item():.4f}")

save_file(student.state_dict(), "model.safetensors")
