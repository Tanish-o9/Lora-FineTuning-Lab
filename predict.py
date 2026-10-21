"""
Saved LoRA adapter ko base model par lagake prediction chalata hai.

Examples:
    python predict.py --adapter runs/lora_r8_q_lin-v_lin "i am so happy today" "this is terrifying"
    python predict.py --adapter runs/lora_r8_q_lin-v_lin --merge "i miss her so much"

--merge: adapter ko base weights mein mila deta hai (merge_and_unload).
         Isse inference mein extra adapter computation nahi hoti.
"""
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="distilbert-base-uncased")
    p.add_argument("--adapter", required=True)
    p.add_argument("--merge", action="store_true")
    p.add_argument("texts", nargs="+")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(LABELS))
    model = PeftModel.from_pretrained(base, args.adapter)
    if args.merge:
        model = model.merge_and_unload()
    model.to(device).eval()

    inputs = tok(args.texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1).cpu()

    for text, pr in zip(args.texts, probs):
        top = int(pr.argmax())
        print(f"{text!r} -> {LABELS[top]} ({pr[top]:.2f})")


if __name__ == "__main__":
    main()
