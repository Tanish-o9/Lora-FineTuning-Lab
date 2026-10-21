#!/usr/bin/env bash
# Poora experiment set: baseline + rank ablation + target-module ablation
# Run: bash run_experiments.sh
set -e

# 1) Baseline: full fine-tuning
python lora_lab.py --mode full

# 2) Rank ablation (alpha = 2 * rank rakha hai taaki scaling same rahe)
for r in 4 8 16 32; do
  python lora_lab.py --mode lora --rank $r --alpha $((2 * r))
done

# 3) Target modules ablation (rank 8): sirf attention ke q,v  vs  poora attention  vs  attention + FFN
python lora_lab.py --mode lora --rank 8 --alpha 16 --targets q_lin k_lin v_lin out_lin
python lora_lab.py --mode lora --rank 8 --alpha 16 --targets q_lin k_lin v_lin out_lin lin1 lin2

echo "Done. Ab chalao: streamlit run app_lora.py"
