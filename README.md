# Godot Micro-Brain (SmolLM2-135M + LoRA)

Godot 4.7.2 ke liye local micro-action model — chhote Godot kaam chunta hai
(JSON tool-call), bade/complex kaam escalation bhejta hai.

## Model
- Base: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Type: LoRA SFT (r=8, alpha=16, dropout=0.05, target=all-linear)
- Precision: FP16 (T4 par train hua)

## Training
- GPU: Tesla T4 (CUDA 12.8, PyTorch 2.8.0+cu128)
- Dataset: public_dataset/godot_microbrain_v1 (12 shards concat, numeric order) — **6000 examples, NO split**
- Epochs: 10 | LR: 0.0002 | Batch: 8 x 8 grad-accum (effective 64)
- Final train loss: 0.19281517267227172
- Duration: 28.1 min
- Split: NONE — trained on all 6000 (user spec 28-09)

## Files
- `adapter_config.json`, `adapter_model.safetensors` — LoRA adapter
- `tokenizer_config.json`, `tokenizer.json` — tokenizer
- `training_report.json` — poora training report
- `TRAINING_COMPLETE` — training verify marker

## Use kaise karein
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct",
                                            dtype=torch.float16, device_map="cuda")
model = PeftModel.from_pretrained(base, ".")   # is repo ka folder
tok = AutoTokenizer.from_pretrained(".")
model.eval()

DEVELOPER = (
    "DEVELOPER\n"
    "You are a local Godot 4.7.2 micro-action model. Return exactly one JSON action. "
    "Use only supplied Godot AI tool names. For complex, ambiguous, multimodal, "
    "architectural, or uncertain work, return an escalation action instead of guessing."
)
prompt = DEVELOPER + "\nUSER: Please inspect the live Godot 4.7.2 API for Node3D.\nASSISTANT:"
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=80, do_sample=False,
                         pad_token_id=tok.pad_token_id)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Verify (sha256)
- `TRAINING_COMPLETE`: `28a99c6df9bfe926d92c047d859ce5d67f02d152133cc1ac5a18ee1b28575fd2`
- `adapter_config.json`: `ed53a79ed2816090125a867c89f641c110642d417fcccea43f4a0cb5789f224b`
- `adapter_model.safetensors`: `45f51efce1774941434f8ce85c56dc38713b5b1a158c092a0ead8d96d642aa8e`
- `base_model.txt`: `b68b303bbb4c05510750c8b422bc102354470146272efe53a546844de91fa517`
- `tokenizer.json`: `bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4`
- `tokenizer_config.json`: `3e9c0d0aff40796f26a2986e6ed1b5e03646765272eb20bacd2492bd020e6c98`
- `training_metrics.json`: `2d3a579d246c56e5ecd794d2446de920643e95585b873f4e9ffc69e03181273a`
- `training_report.json`: `03f1d86339df0e29bc32047c924a72d749b3342cc21d7831bdd837d179d6f251`
