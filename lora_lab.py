"""
LoRA Fine-Tuning Lab - training + evaluation script

Ek hi model ko 2 tareeke se fine-tune karta hai aur compare karta hai:
  --mode full : poora model train (baseline)
  --mode lora : base model frozen, sirf chhote LoRA adapters train

Task  : emotion classification (6 classes) on dair-ai/emotion
Model : distilbert-base-uncased (chhota, T4 GPU par fast)

Examples:
    python lora_lab.py --mode full
    python lora_lab.py --mode lora --rank 8 --alpha 16
    python lora_lab.py --mode lora --rank 16 --alpha 32 --targets q_lin k_lin v_lin out_lin

Har run lora_results.csv mein ek row jodta hai (dashboard: app_lora.py).
"""
import argparse
import random
import time
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

NUM_LABELS = 6  # sadness, joy, love, anger, fear, surprise


# Helpers

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--mode", choices=["full", "lora"], default="lora")

    # LoRA settings
    p.add_argument("--rank", type=int, default=8, help="LoRA rank r")
    p.add_argument("--alpha", type=int, default=16, help="LoRA alpha (scaling = alpha / r)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--targets", nargs="+", default=["q_lin", "v_lin"],
                   help="Kin layers mein LoRA lagana hai (DistilBERT: q_lin k_lin v_lin out_lin lin1 lin2)")

    # Training settings
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="Default: lora=5e-4, full=5e-5")
    p.add_argument("--train_size", type=int, default=8000)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--results", default="lora_results.csv")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


# Data

def build_data(tok, args):
    ds = load_dataset("dair-ai/emotion")
    train = ds["train"].shuffle(seed=args.seed).select(range(min(args.train_size, len(ds["train"]))))
    test = ds["test"]

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    train = train.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    test = test.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")

    collator = DataCollatorWithPadding(tok)  # batch ke andar dynamic padding
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test, batch_size=64, shuffle=False, collate_fn=collator)
    return train_loader, test_loader


# Model

def build_model(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=NUM_LABELS)

    if args.mode == "full":
        return model  # saare parameters trainable

    # Classification head naya (random) hai, isliye usse bhi train karna zaroori hai.
    # modules_to_save = "in modules ko poora train karo aur adapter ke saath save karo".
    heads = [n for n in ("pre_classifier", "classifier", "score") if hasattr(model, n)]

    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.targets,
        modules_to_save=heads,
    )
    return get_peft_model(model, cfg)  # base weights freeze ho jaate hain


# Train + Evaluate

def train(model, loader, args, lr, device):
    use_amp = device == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]  # sirf trainable
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  avg loss {running / len(loader):.4f}")


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    acc = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="macro")["f1"]
    return acc, f1


# Main

def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = args.lr or (5e-4 if args.mode == "lora" else 5e-5)
    print(f"device={device}  mode={args.mode}  lr={lr}")

    tok = AutoTokenizer.from_pretrained(args.model)
    train_loader, test_loader = build_data(tok, args)
    model = build_model(args).to(device)

    trainable, total = count_params(model)
    print(f"trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    # training (time + peak memory)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    t0 = time.perf_counter()
    train(model, train_loader, args, lr, device)
    sync(device)
    train_time = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else float("nan")

    # evaluation
    t0 = time.perf_counter()
    acc, f1 = evaluate_model(model, test_loader, device)
    sync(device)
    eval_time = time.perf_counter() - t0

    # save (LoRA mein sirf adapter save hota hai)
    if args.mode == "lora":
        run_name = f"lora_r{args.rank}_{'-'.join(args.targets)}"
    else:
        run_name = "full"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    saved_mb = dir_size_mb(out)  # tokenizer save karne se PEHLE naapo
    tok.save_pretrained(out)

    row = {
        "run": run_name,
        "mode": args.mode,
        "model": args.model,
        "rank": args.rank if args.mode == "lora" else np.nan,
        "alpha": args.alpha if args.mode == "lora" else np.nan,
        "targets": "+".join(args.targets) if args.mode == "lora" else "all",
        "lr": lr,
        "epochs": args.epochs,
        "train_size": args.train_size,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(trainable / total * 100, 3),
        "train_time_s": round(train_time, 1),
        "peak_train_mem_mb": round(peak_mb, 1),
        "saved_size_mb": round(saved_mb, 2),
        "eval_time_s": round(eval_time, 2),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
    }

    df = pd.DataFrame([row])
    results = Path(args.results)
    df.to_csv(results, mode="a", header=not results.exists(), index=False)
    print("\n" + df.T.to_string(header=False))
    print(f"\nSaved to {out}  |  row appended to {results}")


if __name__ == "__main__":
    main()
