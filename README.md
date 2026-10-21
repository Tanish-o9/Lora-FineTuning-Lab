# Lora-FineTuning-Lab
Benchmarking full fine-tuning vs LoRA on DistilBERT for emotion classification. Compares trainable parameters, training time, memory, checkpoint size and accuracy, with rank and target-module ablations. Built with PyTorch, PEFT and a Streamlit dashboard.

## What this project does

- Fine-tunes `distilbert-base-uncased` on `dair-ai/emotion` (sadness, joy, love, anger, fear, surprise)
- Compares **full fine-tuning (baseline)** against **LoRA adapters** with a frozen base model
- Runs ablations over LoRA **rank**, **alpha** and **target modules**
- Saves reusable adapters and provides a prediction script
- Shows every run in a Streamlit dashboard, compared against the baseline

## Metrics

| Metric | Meaning |
|---|---|
| Trainable params (%) | Share of the model that is actually trained |
| Model memory (MB) | Memory taken by the weights |
| Peak training memory (MB) | Highest GPU memory during training |
| Train time (s) | Wall-clock time for all epochs |
| Saved size (MB) | Full checkpoint vs adapter-only checkpoint |
| Accuracy / macro-F1 | Quality on the test split |

## Project structure

```
lora_lab.py          # training + evaluation (--mode full | lora)
predict.py           # load a saved adapter and predict
app_lora.py          # Streamlit dashboard
run_experiments.sh   # baseline + rank and target-module ablations
requirements_lora.txt
lora_results.csv     # created after the first run (one row per run)
runs/                # saved adapters / checkpoints
```

## Setup

```bash
pip install -r requirements_lora.txt
```

A GPU is recommended (Google Colab free T4 is enough).

## Usage

Single runs:

```bash
python lora_lab.py --mode full
python lora_lab.py --mode lora --rank 8 --alpha 16
python lora_lab.py --mode lora --rank 16 --alpha 32 --targets q_lin k_lin v_lin out_lin
```

All experiments (baseline, rank 4/8/16/32, target-module ablation):

```bash
bash run_experiments.sh
```

Predict with a saved adapter:

```bash
python predict.py --adapter runs/lora_r8_q_lin-v_lin "i am so happy today" "this is terrifying"
python predict.py --adapter runs/lora_r8_q_lin-v_lin --merge "i miss her so much"
```

Dashboard:

```bash
streamlit run app_lora.py
```

## Key settings

| Argument | Default | Notes |
|---|---|---|
| `--mode` | `lora` | `full` or `lora` |
| `--rank` | 8 | LoRA rank `r` |
| `--alpha` | 16 | Scaling is `alpha / r` |
| `--targets` | `q_lin v_lin` | Layers that receive LoRA |
| `--lr` | 5e-4 (LoRA), 5e-5 (full) | LoRA needs a higher learning rate |
| `--train_size` | 8000 | Training examples used |
| `--seed` | 42 | Fixed for reproducibility |

## Results

> Replace with your own measured numbers. Run each setting with 2-3 seeds and report the mean.

| Run | Trainable % | Saved size (MB) | Peak train mem (MB) | Train time (s) | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| Full fine-tuning | 100 | X | X | X | X | X |
| LoRA r=8 (q, v) | X | X | X | X | X | X |
| LoRA r=16 (q, v) | X | X | X | X | X | X |
| LoRA r=8 (all attention) | X | X | X | X | X | X |

**Hardware / versions:** GPU = X, torch = X, transformers = X, peft = X

## Findings

- LoRA trained X% of the parameters and reduced checkpoint size from X MB to X MB.
- Accuracy changed by X points compared with full fine-tuning.
- Best rank / target modules: X, because X.

## Notes and limitations

- LoRA reduces **trainable parameters and adapter storage**. It does not shrink the base model and does not automatically speed up inference (merging the adapter removes the extra computation but the model stays the same size).
- DistilBERT is small, so memory and time differences are modest. The benefit grows with larger models.
- The classification head is new, so it is trained fully and saved with the adapter. Adapter size therefore includes the head.
- Small accuracy differences can be noise; use several seeds before drawing conclusions.
- Quality is measured on a single dataset and task.

## Tech stack

Python, PyTorch, Hugging Face Transformers, PEFT, Datasets, Evaluate, Streamlit
