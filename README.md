# Fine-tuning Qwen2.5-Math with QLoRA: V1 vs V4

I fine-tuned **Qwen2.5-Math-1.5B-Instruct** on the GSM8K mathematics word-problem dataset to see whether supervised fine-tuning would improve its ability to solve unseen questions. This repository documents two experiments: **V1**, using LoRA on two attention projections, and **V4**, using LoRA on seven attention/feed-forward projections.

**The main finding:** V4 achieved a lower validation loss, but its accuracy on 200 GSM8K test questions was worse than the original model. A lower token-prediction loss does not necessarily mean better problem-solving accuracy.

## Results

| Model | LoRA target modules | Trainable parameters | Validation loss | Correct / 200 | Accuracy |
|---|---|---:|---:|---:|---:|
| Original Qwen | None | — | — | 163 | **81.5%** |
| V1 | `q_proj`, `v_proj` | 1,089,536 | 0.5417 | 163 | **81.5%** |
| V4 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | 9,232,384 | **0.4935** | 138 | **69.0%** |

Compared with the original model, V4 fixed 17 questions but regressed on 42. Both models answered 121 questions correctly and missed 20. The original model had 7 unextracted answers; V4 had 9. Inspecting incorrect outputs showed genuine mathematical mistakes as well as one repetitive generation.

These numbers come from the completed experiments; the notebook's training outputs have been cleared for a clean GitHub presentation.

## Repository contents

```text
qwen-math-qlora-v1-v4/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── qwen_math_qlora_v1_v4.ipynb
```

## How to run

1. Create a Python environment with a CUDA-enabled PyTorch installation compatible with your NVIDIA GPU.
2. Install the other dependencies: `pip install -r requirements.txt`.
3. Open `notebooks/qwen_math_qlora_v1_v4.ipynb` in Jupyter or VS Code.
4. Run the cells in order. Training V1 and V4 takes time; each experiment loads a fresh quantized base model.

The experiments were developed on an **RTX 3070 Laptop GPU with 8 GB VRAM**, using **4-bit NF4 quantization** and **LoRA adapters**. The notebook downloads the base model and GSM8K dataset from Hugging Face when run. Model checkpoints and generated predictions are intentionally not included in this repository.

## Experimental design and limitations

- Dataset: `openai/gsm8k` (`main`), with 90% of its training split used for training and 10% for validation (seed 42).
- Evaluation: first 200 questions of the official GSM8K test set, using greedy decoding and numerical answer extraction.
- V1: 2 epochs; batch size 1, gradient accumulation 8; learning rate `2e-4`.
- V4: 2 epochs; batch size 2, gradient accumulation 4; learning rate `2e-4`, 50 warmup steps.
- V1 and V4 changed **both** LoRA module coverage and batch settings; this is not a perfectly controlled single-variable experiment.
- Training asked for `Final answer: <number>` while evaluation asked for a boxed answer. The extractor handled both formats, but the prompt mismatch is a limitation.
- Results on 200 questions are useful for a small comparison, not a definitive benchmark.

## Where are the trained models?

This repository contains the **training and evaluation code**, not the full pretrained model or LoRA adapter weights. The base model is available at [Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct). If I publish my adapters separately, I'll link them here.

## What I learned

This project helped me practice quantization, parameter-efficient fine-tuning, chat-template tokenization, label masking, dynamic padding, and paired error analysis. Most importantly, it showed me why mathematical accuracy should be measured directly instead of relying on validation loss alone.
