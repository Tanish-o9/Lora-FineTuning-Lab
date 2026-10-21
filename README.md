# 🧪 LoRA Fine-Tuning Lab

A practical, production-grade benchmarking and experimentation lab comparing **Full Fine-Tuning** with **LoRA (Low-Rank Adaptation)** on Hugging Face Transformer models.

This lab evaluates and logs key efficiency trade-offs across **Trainable Parameters**, **Peak GPU Memory (VRAM)**, **Training Time**, **Disk Checkpoint Size**, and **Downstream Model Accuracy / F1-Score**.

---

## 📌 Project Overview

Fine-tuning large language and transformer models usually requires updating all model weights (**Full Fine-Tuning**), which demands significant GPU VRAM and produces multi-gigabyte checkpoints for each downstream task.

**LoRA (Low-Rank Adaptation - Hu et al., 2021)** freezes the pre-trained model weights and injects trainable rank-decomposition matrices into Transformer attention layers:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

Where:
- $W_0 \in \mathbb{R}^{d \times k}$ is the frozen pre-trained weight matrix.
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are low-rank adapter matrices with rank $r \ll \min(d, k)$.
- $\alpha$ is a constant scaling factor.

### Task & Model Specifications
- **Task:** 6-Class Emotion Classification (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`)
- **Dataset:** `dair-ai/emotion` (Hugging Face Datasets)
- **Base Model:** `distilbert-base-uncased` (66M parameters)
- **Hardware Benchmarked On:** NVIDIA GeForce RTX 2050 (4GB VRAM) | PyTorch `2.6.0+cu124` with CUDA FP16 AMP

---

## 📊 Real Tested Benchmark Results

The following table reflects **actual benchmarked data** produced directly by [lora_lab.py](file:///c:/Users/tanis/Projects/LoRA%20Fine-Tuning%20Lab/lora_lab.py) and recorded in [lora_results.csv](file:///c:/Users/tanis/Projects/LoRA%20Fine-Tuning%20Lab/lora_results.csv):

| Run Name | Mode | Rank ($r$) | Alpha ($\alpha$) | Target Modules | Trainable Params | Trainable % | Peak VRAM | Training Time | Saved Size | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`full` (Baseline)** | Full | — | — | All Layers | 66,958,086 | **100.0%** | 1,324.5 MB | 34.9 s | **255.44 MB** | **90.30%** | **0.8382** |
| **`lora_r8_q_lin-v_lin`** | LoRA | 8 | 16 | `q_lin` + `v_lin` | 742,662 | **1.097%** | 684.1 MB | 16.6 s | **2.84 MB** | **81.15%** | **0.7363** |
| **`lora_r16_all_attn`** | LoRA | 16 | 32 | `q, k, v, out` | 1,185,030 | **1.739%** | 743.1 MB | 19.2 s | **4.53 MB** | **85.85%** | **0.8010** |

---

## 🔍 Key Findings & Comparative Reductions

Comparing **LoRA ($r=8$)** directly against **Full Fine-Tuning**:

1. **Storage Reduction (-98.89%):**
   - Full model checkpoint: **255.44 MB**
   - LoRA adapter checkpoint: **2.84 MB**
   - *Advantage:* You can store dozens of specialized task adapters in the storage space required for a single full model.

2. **Parameter Efficiency (-98.9% Trainable Parameters):**
   - Full Fine-Tuning trains **66,958,086** parameters.
   - LoRA ($r=8$) trains only **742,662** parameters (including the newly initialized classification head).

3. **Memory & VRAM Savings (-48.35% Peak Memory):**
   - Peak training VRAM drops from **1,324.5 MB** down to **684.1 MB**.
   - Because 99% of parameters are frozen, the AdamW optimizer state footprint is drastically reduced.

4. **Training Speedup (2.1x Faster):**
   - Full model epoch training: **34.9 seconds**.
   - LoRA epoch training: **16.6 seconds**.

5. **Target Modules Ablation:**
   - Adapting only Query and Value projections (`q_lin`, `v_lin`) achieves **81.15%** accuracy.
   - Expanding LoRA to all attention projections (`q_lin`, `k_lin`, `v_lin`, `out_lin`) with $r=16$ boosts accuracy to **85.85%** (Macro F1: **0.8010**) while still using under **1.8%** of total parameters.

---

## 🎯 Verified Inference Output

Using [predict.py](file:///c:/Users/tanis/Projects/LoRA%20Fine-Tuning%20Lab/predict.py) with the trained adapter:

```bash
python predict.py --adapter runs/lora_r8_q_lin-v_lin "i am so thrilled and happy today" "i feel terribly scared"
```

**Real Output:**
```text
'i am so thrilled and happy today' -> joy (confidence: 0.93)
'i feel terribly scared'           -> fear (confidence: 0.98)
```

With `--merge` (merges $W_0 + \Delta W$ into the base weights for zero-latency inference):

```bash
python predict.py --adapter runs/lora_r8_q_lin-v_lin --merge "this made me super furious" "i miss my family"
```

**Real Output:**
```text
'this made me super furious' -> anger (confidence: 0.96)
'i miss my family'           -> sadness (confidence: 0.93)
```

---

## 📂 Repository Structure

```
LoRA Fine-Tuning Lab/
├── lora_lab.py              # Main training, evaluation, & benchmarking script
├── predict.py              # CLI inference script (supports standalone & --merge)
├── app_lora.py             # Streamlit visual dashboard for interactive metrics
├── run_experiments.sh      # Bash script to run full ablation sweeps
├── lora_results.csv        # Real recorded benchmark results
├── requirements_lora.txt   # Python dependency list
├── runs/                   # Checkpoints directory
│   ├── full/               # Full model checkpoint (~255 MB)
│   ├── lora_r8_.../        # LoRA adapter checkpoint (~2.8 MB)
│   └── lora_r16_.../       # LoRA adapter checkpoint (~4.5 MB)
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements_lora.txt
```

*(Recommended: PyTorch with CUDA support if training on GPU).*

---

### 2. Running Fine-Tuning Experiments

#### Run Full Fine-Tuning (Baseline):
```bash
python lora_lab.py --mode full --epochs 2 --train_size 4000
```

#### Run LoRA Fine-Tuning (Standard Attention: Query + Value):
```bash
python lora_lab.py --mode lora --rank 8 --alpha 16 --targets q_lin v_lin --epochs 2 --train_size 4000
```

#### Run LoRA with All Attention Projections:
```bash
python lora_lab.py --mode lora --rank 16 --alpha 32 --targets q_lin k_lin v_lin out_lin --epochs 2 --train_size 4000
```

#### CLI Parameters Reference:

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--model` | str | `distilbert-base-uncased` | Base Hugging Face model |
| `--mode` | str | `lora` | Training mode (`full` or `lora`) |
| `--rank` | int | `8` | LoRA rank dimension ($r$) |
| `--alpha` | int | `16` | LoRA alpha scaling factor ($\alpha$) |
| `--dropout` | float | `0.1` | LoRA dropout probability |
| `--targets` | list | `["q_lin", "v_lin"]` | Module layers to inject LoRA adapters into |
| `--epochs` | int | `3` | Number of training epochs |
| `--batch_size` | int | `32` | Training batch size |
| `--lr` | float | Auto | Learning rate (`5e-4` for LoRA, `5e-5` for Full) |
| `--train_size` | int | `8000` | Number of training samples to load |
| `--max_len` | int | `64` | Maximum sequence token length |
| `--out_dir` | str | `runs` | Output directory for checkpoints |
| `--results` | str | `lora_results.csv` | CSV file for logging benchmark metrics |

---

### 3. Running Predictions / Inference

To run inference on custom text using any trained adapter:

```bash
python predict.py --adapter runs/lora_r8_q_lin-v_lin "Today has been the best day ever!"
```

To permanently fuse the adapter matrices into the base model weights (via `merge_and_unload()`) for faster inference:

```bash
python predict.py --adapter runs/lora_r8_q_lin-v_lin --merge "I am terrified of heights"
```

---

### 4. Interactive Streamlit Dashboard

Launch the visual dashboard to analyze and compare metrics across all runs:

```bash
python -m streamlit run app_lora.py
```

The upgraded dashboard features 7 interactive modules:
1. **🚀 In-Browser Training Studio:** Train LoRA adapters or full models directly from the UI using custom CSV files or Hugging Face datasets (`fancyzhx/ag_news`, `dair-ai/emotion`, `stanfordnlp/imdb`).
2. **📜 Lab Activity & History Timeline:** Complete chronological audit trail of all experiments with timestamps, hyperparameters, accuracy, compute duration, checkpoint size, and file explorer.
3. **📊 Analytics & Multi-Dimensional Radar:** 5-axis spider chart comparing Storage Efficiency, VRAM Savings, Speedup, Accuracy, and F1 across runs alongside Pareto frontier curves.
4. **⚔️ Model Arena (Side-by-Side):** Test any two models head-to-head on the same input sentence to compare predictions, confidence distributions, and inference latency.
5. **🧪 Live Inference Playground:** Real-time text classification with confidence gauge meters, animated probability bars, and instant class tagging.
6. **🧮 LoRA Parameter Calculator:** Interactive sliders for Rank ($r$), Alpha ($\alpha$), and projection targets with real-time parameter and memory scaling estimation.
7. **📋 Benchmark Database & Export:** Full benchmark dataset and baseline percentage reduction matrix with instant CSV download.

---

## 💡 Engineering Highlights

- **Dynamic Padding:** Uses `DataCollatorWithPadding` so sequences are padded dynamically per batch, minimizing wasted computation and memory.
- **AMP Mixed Precision:** Uses `torch.autocast` with `GradScaler` (FP16) on CUDA for maximum throughput.
- **Head Adaptation (`modules_to_save`):** DistilBERT's classification head is newly initialized for the 6-class task; `LoraConfig(modules_to_save=[...])` ensures the head is trained and saved alongside adapter weights.
- **Adaptive Learning Rates:** Automatically applies `5e-4` for LoRA (faster adaptation for new adapter weights) and `5e-5` for full fine-tuning (preventing catastrophic forgetting).
