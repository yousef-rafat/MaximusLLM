import numpy as np
import matplotlib.pyplot as plt
import lm_eval
from lm_eval.utils import make_table
from model import Model, Config
from transformers import AutoTokenizer
from safetensors.torch import load_file
from lm_eval.models.huggingface import HFLM
from huggingface_hub import hf_hub_download
import torch

MODELS = {
    "MaximusLLM (0.19B)": "yousefg/MaximusLLM", 
    "Qwen2.5 (0.5B)": "Qwen/Qwen2.5-0.5B",
    "SmolLM2 (360M)": "HuggingFaceTB/SmolLM2-360M",
    "OPT (350M)": "facebook/opt-350m"
}

TASKS = ["arc_easy", "hellaswag", "piqa", "boolq"]

DEVICE = "cuda:0"
BATCH_SIZE = "auto"

class HFWrapper(torch.nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.base_model(input_ids, attention_mask=attention_mask)
            
        if isinstance(out, torch.Tensor):
            class DummyOutput:
                def __init__(self, logits):
                    self.logits = logits
            return DummyOutput(out)
        return out

def run_evaluations():
    all_results = {}
    
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name} ({model_id})...")
        print(f"{'='*50}\n")

        if model_name == "MaximusLLM (0.19B)":
            config = Config.from_pretrained(model_id)
            model = Model(config, DEVICE)
            ckpt = load_file(hf_hub_download(model_id, "model.safetensors"))
            model.load_state_dict(ckpt)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = HFWrapper(model, config)

            lm_obj = HFLM(
                pretrained=model, 
                tokenizer=tokenizer, 
                batch_size=BATCH_SIZE
            )

            results = lm_eval.simple_evaluate(
                model=lm_obj,
                tasks=TASKS,
                num_fewshot=0,
            )

        else:    
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_id}",
                tasks=TASKS,
                num_fewshot=0,
                device=DEVICE,
                batch_size=BATCH_SIZE,
            )
        
        print(f"\nResults for {model_name}:")
        print(make_table(results))
        
        model_metrics = {}
        for task in TASKS:
            task_data = results["results"].get(task, {})
            acc = task_data.get("acc_norm,none", task_data.get("acc,none", 0.0))
            model_metrics[task] = acc * 100 
            
        all_results[model_name] = model_metrics
        
    return all_results

def plot_results(all_results):
    print("\nGenerating visualization...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    _, ax = plt.subplots(figsize=(12, 7))
    
    model_names = list(MODELS.keys())
    task_names = TASKS
    
    x = np.arange(len(task_names))
    width = 0.8 / len(model_names)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)]
    
    for i, model_name in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        scores = [all_results[model_name][task] for task in task_names]
        
        bars = ax.bar(x + offset, scores, width, label=model_name, color=colors[i], edgecolor='white')
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    ax.set_ylabel('Accuracy / Normalized Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Zero-Shot Benchmark Comparison (0.2B - 0.6B Parameters)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    display_names = [t.replace("_", " ").title() for t in task_names]
    ax.set_xticklabels(display_names, fontsize=12, fontweight='bold')
    
    ax.legend(title='Models', title_fontsize='12', fontsize='11', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, zorder=0)
    ax.text(-0.5, 50.5, 'Random Baseline (Binary)', color='gray', fontsize=9)

    plt.tight_layout()
    
    plt.savefig("benchmark_comparison.png", dpi=300, bbox_inches='tight')
    print("Plot saved as 'benchmark_comparison.png'")
    plt.show()

if __name__ == "__main__":
    results = run_evaluations()
    plot_results(results)
