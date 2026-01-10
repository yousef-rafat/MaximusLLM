import time
import torch
from datasets import load_dataset
from torch.src.infer import general_generate_fn

def benchmark_forward(model, input_sample, attention_mask, device="cpu", runs=20): # for speed/performance
    model = model.to(device)
    input_sample = input_sample.to(device)
    attention_mask = attention_mask.to(device)
    model.eval()

    torch.set_grad_enabled(False)

    # warmup
    for _ in range(3):
        _ = model(input_sample, attention_mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.time()
        _ = model(input_sample, attention_mask)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    avg = sum(times) / len(times)
    return avg

def general_benchmark_fn(model, tokenizer, prompt):
    inputs = tokenizer.encode(prompt)
    if hasattr(model, "generate"):
        outputs = model.generate(*inputs, do_sample = False)
    else:
        outputs = general_generate_fn(model, inputs, tokenizer.eos_token)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text.strip().lower()

# ACCURACY BASED
def bench_mark_glue(model, model2, tokenizer):
    dataset = load_dataset("aps/super_glue", split="train", streaming = True)
    id2label = {0: "yes", 1: "no"}
    correct_model_1 = 0
    correct_model_2 = 0
    for batch in dataset:
        sentence1, sentence2 = batch["sentence1"], batch["sentence2"]
        prompt = (
            f"Premise: {sentence1}\n"
            f"Hypothesis: {sentence2}\n"
            f"Question: Does the premise imply the hypothesis? Yes or No?\n"
            f"Answer:"
        )
        label_str = id2label[batch["label"]]
        out1 = general_benchmark_fn(model, tokenizer, prompt)
        out2 = general_benchmark_fn(model2, tokenizer, prompt)

        if out1.startswith(label_str):
            correct_model_1 += 1
        if out2.startswith(label_str):
            correct_model_2 += 1

    print(f"Total correct answers for model 1: {correct_model_1}")
    print(f"Total correct answers for model 2: {correct_model_2}")


if __name__ == "__main__":
    model1 = ...
    model2 = ...

    dummy = torch.randint(0, 30000, (1, 4_000))
    attention_mask = torch.ones_like(dummy)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t1 = benchmark_forward(model1, dummy, attention_mask, device)
    t2 = benchmark_forward(model2, dummy, attention_mask, device)

    print(f"Model 1 avg time: {t1*1000:.3f} ms")
    print(f"Model 2 avg time: {t2*1000:.3f} ms")
    print(f"Speedup: {t2/t1:.2f}x (model1 vs model2)")
