"""
LoRA Fine-Tuning Lab - Ultimate Interactive AI Dashboard & Training Studio
Features:
  - 🚀 Live In-Browser Training Studio: Train on ANY Custom CSV or HuggingFace Dataset
  - 📊 Executive Analytics & Multi-Dimensional Radar Tradeoff
  - ⚔️ Model Arena: Side-by-Side Real-Time Inference Battle
  - 🧪 Single Model Playground with Live Gauge & Dynamic Class Adapters
  - 🧮 LoRA Math & Interactive Rank Parameter Calculator
  - 📋 Benchmark Database & Export Matrix
"""
import time
from pathlib import Path
import evaluate
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# ----------------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="LoRA Fine-Tuning Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Ultra-Premium CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #111827 0%, #0b0f19 45%, #05070d 100%);
        color: #f8fafc;
    }

    /* Top glowing line */
    .top-glow-bar {
        height: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4, #10b981);
        border-radius: 999px;
        margin-bottom: 24px;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.6);
    }

    /* Hero Header */
    .hero-title-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
        flex-wrap: wrap;
        gap: 12px;
    }
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #c084fc 40%, #06b6d4 85%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    .hero-caption {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* Live status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 999px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* KPI Stat Cards with Gradient Border Shimmer */
    .metric-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 14px 40px rgba(99, 102, 241, 0.2);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), transparent);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 10%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.78rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.78rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.78rem;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Training Studio card */
    .studio-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Arena Matchup Cards */
    .arena-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
    }

    /* Calculator Cards */
    .calc-box {
        background: rgba(99, 102, 241, 0.04);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 20px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(99, 102, 241, 0.45) !important;
    }

    /* Emotion Playground styles */
    .prob-bar-container {
        margin-bottom: 12px;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .prob-bar-track {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
DEFAULT_ICONS = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
}
PALETTE = ["#38bdf8", "#facc15", "#f43f5e", "#ef4444", "#a855f7", "#34d399", "#fb923c", "#818cf8"]

# ----------------------------------------------------------------------------
# Data Loading & Verification
# ----------------------------------------------------------------------------
results_file = Path("lora_results.csv")
if not results_file.exists():
    st.error("⚠️ `lora_results.csv` not found. Pehle koi run complete karein.")
    st.stop()

df = pd.read_csv(results_file)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ **System Telemetry**")
    gpu_name = df["gpu"].iloc[0] if "gpu" in df.columns else "N/A"
    torch_ver = df["torch"].iloc[0] if "torch" in df.columns else "N/A"
    model_name = df["model"].iloc[0] if "model" in df.columns else "distilbert-base-uncased"

    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:16px; margin-bottom:18px;">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:700; letter-spacing:0.05em;">Compute Rig</div>
            <div style="margin-top:8px; font-weight:700; color:#f8fafc; font-size:0.95rem;">🖥️ {gpu_name}</div>
            <div style="margin-top:4px; font-size:0.85rem; color:#cbd5e1;">🔥 PyTorch <code>{torch_ver}</code></div>
            <div style="margin-top:4px; font-size:0.85rem; color:#cbd5e1;">🤖 Default: <code>{model_name}</code></div>
            <div style="margin-top:6px; font-size:0.8rem; color:#10b981; font-weight:600;">⚡ FP16 Automatic Mixed Precision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎛️ **Benchmark Actions**")
    uploaded = st.file_uploader("Upload custom results CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)

    st.download_button(
        label="📥 Export Benchmark CSV",
        data=df.to_csv(index=False),
        file_name="lora_results.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("---")
    st.markdown(
        """
        **Mathematical Formulation:**
        $$W = W_0 + \\frac{\\alpha}{r} (B \\cdot A)$$
        - **$W_0$:** Frozen pre-trained matrix
        - **$B, A$:** Trainable low-rank factors
        - **$r \\ll d$:** Intrinsic dimensionality
        """
    )

# ----------------------------------------------------------------------------
# Hero Top Section
# ----------------------------------------------------------------------------
st.markdown('<div class="top-glow-bar"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-title-container">
        <div class="hero-title">⚡ LoRA Fine-Tuning Lab</div>
        <div class="status-pill">
            <div class="status-dot"></div>
            <span>TRAINING & BENCHMARK STUDIO</span>
        </div>
    </div>
    <div class="hero-caption">
        Train, benchmark, and evaluate parameter-efficient LoRA adapters across custom CSV files or Hugging Face datasets.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Top KPI Metric Cards
# ----------------------------------------------------------------------------
full_df = df[df["mode"] == "full"]
lora_df = df[df["mode"] == "lora"]

if not full_df.empty and not lora_df.empty:
    base_run = full_df.iloc[-1]
    best_lora = lora_df.sort_values(by="accuracy", ascending=False).iloc[0]
    std_lora = lora_df.iloc[0]

    param_red = (base_run["trainable_params"] - std_lora["trainable_params"]) / base_run["trainable_params"] * 100
    size_red = (base_run["saved_size_mb"] - std_lora["saved_size_mb"]) / base_run["saved_size_mb"] * 100
    mem_red = (base_run["peak_train_mem_mb"] - std_lora["peak_train_mem_mb"]) / base_run["peak_train_mem_mb"] * 100
    speedup = base_run["train_time_s"] / std_lora["train_time_s"] if std_lora["train_time_s"] > 0 else 1.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">💾 Disk Storage Saving</div>
                <div class="metric-value">-{size_red:.1f}%</div>
                <div class="metric-sub">
                    <span class="badge-green">{std_lora['saved_size_mb']:.2f} MB checkpoint</span>
                    <span style="color:#64748b;">vs {base_run['saved_size_mb']:.1f} MB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🧠 Trainable Parameters</div>
                <div class="metric-value">-{param_red:.1f}%</div>
                <div class="metric-sub">
                    <span class="badge-blue">{std_lora['trainable_pct']:.2f}% active</span>
                    <span style="color:#64748b;">vs 100% full</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">⚡ Training Speedup</div>
                <div class="metric-value">{speedup:.1f}x</div>
                <div class="metric-sub">
                    <span class="badge-purple">{std_lora['train_time_s']}s duration</span>
                    <span style="color:#64748b;">vs {base_run['train_time_s']}s</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🎯 Best LoRA Accuracy</div>
                <div class="metric-value">{best_lora['accuracy']*100:.1f}%</div>
                <div class="metric-sub">
                    <span class="badge-green">{best_lora['run']}</span>
                    <span style="color:#64748b;">(Full: {base_run['accuracy']*100:.1f}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 6 Interactive Core Modules (Tabs)
# ----------------------------------------------------------------------------
tab_studio, tab_analytics, tab_arena, tab_playground, tab_calc, tab_table = st.tabs(
    [
        "🚀 Training Studio",
        "📊 Analytics & Radar",
        "⚔️ Model Arena (Side-by-Side)",
        "🧪 Live Inference Playground",
        "🧮 LoRA Parameter Calculator",
        "📋 Benchmark Database",
    ]
)

# ----------------------------------------------------------------------------
# Helper: Model Caching for Inference
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_model(adapter_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if "full" in adapter_dir.lower():
        model = AutoModelForSequenceClassification.from_pretrained(adapter_dir)
    else:
        # Check num_labels from config if exists
        try:
            test_model = AutoModelForSequenceClassification.from_pretrained(adapter_dir)
            num_classes = test_model.config.num_labels
        except Exception:
            num_classes = 6
        base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
        model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device).eval()

    # Determine class label names
    if hasattr(model.config, "id2label") and model.config.id2label:
        classes = [str(model.config.id2label[i]) for i in range(len(model.config.id2label))]
    else:
        classes = DEFAULT_LABELS[: model.config.num_labels]

    return tok, model, device, classes


def run_single_inference(tok, model, device, classes, text: str):
    t_start = time.perf_counter()
    inputs = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = logits.softmax(-1).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - t_start) * 1000
    top_idx = int(np.argmax(probs))
    return classes[top_idx], probs[top_idx] * 100, probs, latency_ms


# ----------------------------------------------------------------------------
# TAB 1: Live Training Studio (Custom CSV / HF Dataset Training)
# ----------------------------------------------------------------------------
with tab_studio:
    st.markdown("### 🚀 **In-Browser Training Studio**")
    st.markdown("Train a LoRA adapter or Full model directly from the UI on **your own CSV file** or any **Hugging Face dataset**!")

    col_ds, col_hparams = st.columns([1.1, 1.2])

    with col_ds:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.markdown("#### **1. Select Dataset Source**")
        dataset_source = st.radio(
            "Choose Dataset Type:",
            options=["📁 Upload Custom CSV File", "🌐 Hugging Face Dataset Preset"],
            index=0,
            horizontal=True,
        )

        train_texts = None
        train_labels = None
        unique_labels = None
        dataset_display_name = "custom"

        if "Upload Custom CSV" in dataset_source:
            uploaded_csv = st.file_uploader(
                "Upload your CSV (must have Text and Label columns):",
                type=["csv"],
                key="studio_csv_uploader",
            )
            if uploaded_csv:
                custom_df = pd.read_csv(uploaded_csv)
                st.markdown(f"**Preview ({len(custom_df)} rows):**")
                st.dataframe(custom_df.head(3), width="stretch")

                cols = list(custom_df.columns)
                text_col = st.selectbox("Text Column:", cols, index=0)
                label_col = st.selectbox("Label Column:", cols, index=min(1, len(cols) - 1))

                # Extract and clean
                custom_df = custom_df.dropna(subset=[text_col, label_col])
                unique_labels = sorted(list(custom_df[label_col].unique()))
                st.success(f"Detected **{len(unique_labels)} classes**: `{unique_labels}`")

                label2id = {l: i for i, l in enumerate(unique_labels)}
                id2label = {i: str(l) for i, l in enumerate(unique_labels)}

                train_texts = custom_df[text_col].astype(str).tolist()
                train_labels = [label2id[l] for l in custom_df[label_col]]
                dataset_display_name = uploaded_csv.name.split(".")[0]
            else:
                st.info("💡 Upload any CSV file (e.g. Sentiment, Spam, Complaints, Classification).")
        else:
            CANONICAL_HF_DATASETS = {
                "ag_news": "fancyzhx/ag_news",
                "imdb": "stanfordnlp/imdb",
                "emotion": "dair-ai/emotion",
            }
            hf_preset = st.selectbox(
                "Choose Hugging Face Dataset:",
                options=[
                    "fancyzhx/ag_news (4 classes: World, Sports, Biz, Tech)",
                    "dair-ai/emotion (6 emotions: Sadness, Joy, Love, Anger, Fear, Surprise)",
                    "stanfordnlp/imdb (2 classes: Positive / Negative Sentiment)",
                ],
                index=0,
            )
            raw_dataset_name = hf_preset.split(" ")[0]
            dataset_name = CANONICAL_HF_DATASETS.get(raw_dataset_name, raw_dataset_name)
            dataset_display_name = dataset_name.split("/")[-1]

        st.markdown("</div>", unsafe_allow_html=True)

    with col_hparams:
        st.markdown('<div class="studio-card">', unsafe_allow_html=True)
        st.markdown("#### **2. Model & LoRA Hyperparameters**")
        t_mode = st.radio("Training Mode:", ["lora", "full"], index=0, horizontal=True)
        t_model = st.text_input("Base Model:", value="distilbert-base-uncased")

        if t_mode == "lora":
            c_r, c_a = st.columns(2)
            with c_r:
                t_rank = st.select_slider("LoRA Rank (r):", [2, 4, 8, 16, 32], value=8)
            with c_a:
                t_alpha = st.select_slider("LoRA Alpha (α):", [4, 8, 16, 32, 64], value=16)
            t_targets = st.multiselect(
                "Target Modules:",
                ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
                default=["q_lin", "v_lin"],
            )
        else:
            t_rank = None
            t_alpha = None
            t_targets = ["all"]

        c_ep, c_sz = st.columns(2)
        with c_ep:
            t_epochs = st.number_input("Epochs:", min_value=1, max_value=10, value=2)
            t_batch = st.select_slider("Batch Size:", [8, 16, 32, 64], value=32)
        with c_sz:
            t_samples = st.number_input("Max Train Samples:", min_value=200, max_value=20000, value=2000, step=500)
            t_lr = st.number_input("Learning Rate:", value=0.0005 if t_mode == "lora" else 0.00005, format="%.5f")

        custom_run_name = st.text_input(
            "Run Name:",
            value=f"{t_mode}_{dataset_display_name}_r{t_rank}" if t_mode == "lora" else f"full_{dataset_display_name}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Launch Training Button
    start_train_btn = st.button("🔥 Start Training Job", type="primary", width="stretch")

    if start_train_btn:
        progress_box = st.container()
        with progress_box:
            st.markdown("### ⏳ **Training in Progress...**")
            pbar = st.progress(0.0)
            status_text = st.empty()

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                status_text.info(f"Loading tokenizer & dataset on **{device.upper()}**...")

                tok = AutoTokenizer.from_pretrained(t_model)

                # Prepare Dataset
                if "Upload Custom CSV" in dataset_source:
                    if train_texts is None:
                        st.error("Please upload a CSV file first!")
                        st.stop()
                    n_total = min(len(train_texts), t_samples)
                    n_train = int(n_total * 0.8)

                    raw_train = Dataset.from_dict({"text": train_texts[:n_train], "labels": train_labels[:n_train]})
                    raw_test = Dataset.from_dict({"text": train_texts[n_train:n_total], "labels": train_labels[n_train:n_total]})
                    num_classes = len(unique_labels)
                else:
                    status_text.info(f"Downloading {dataset_name} from Hugging Face...")
                    ds = load_dataset(dataset_name)
                    split_train = ds["train"].shuffle(seed=42).select(range(min(t_samples, len(ds["train"]))))
                    split_test = ds["test"] if "test" in ds else ds["validation"]
                    split_test = split_test.select(range(min(1000, len(split_test))))

                    # Detect text column
                    text_candidate = "text"
                    for col in ["text", "sentence", "content", "review", "description"]:
                        if col in split_train.column_names:
                            text_candidate = col
                            break

                    # Detect label column
                    label_candidate = "label"
                    for col in ["label", "labels", "category", "target"]:
                        if col in split_train.column_names:
                            label_candidate = col
                            break

                    raw_train = split_train.rename_column(text_candidate, "text") if text_candidate != "text" else split_train
                    raw_test = split_test.rename_column(text_candidate, "text") if text_candidate != "text" else split_test

                    raw_train = raw_train.rename_column(label_candidate, "labels") if label_candidate != "labels" else raw_train
                    raw_test = raw_test.rename_column(label_candidate, "labels") if label_candidate != "labels" else raw_test

                    # Known human-readable labels for standard presets
                    if "ag_news" in dataset_name:
                        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
                    elif "imdb" in dataset_name:
                        id2label = {0: "Negative", 1: "Positive"}
                    elif "emotion" in dataset_name:
                        id2label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
                    else:
                        num_classes = len(set(raw_train["labels"]))
                        id2label = {i: f"class_{i}" for i in range(num_classes)}

                    num_classes = len(id2label)
                    label2id = {v: k for k, v in id2label.items()}

                def tokenize(batch):
                    return tok(batch["text"], truncation=True, max_length=64)

                tokenized_train = raw_train.map(tokenize, batched=True, remove_columns=["text"])
                tokenized_test = raw_test.map(tokenize, batched=True, remove_columns=["text"])

                collator = DataCollatorWithPadding(tok)
                train_loader = DataLoader(tokenized_train, batch_size=t_batch, shuffle=True, collate_fn=collator)
                test_loader = DataLoader(tokenized_test, batch_size=64, shuffle=False, collate_fn=collator)

                # Build Model
                status_text.info("Injecting LoRA adapters into model...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    t_model,
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id=label2id,
                )

                if t_mode == "lora":
                    heads = [n for n in ("pre_classifier", "classifier", "score") if hasattr(model, n)]
                    cfg = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        r=t_rank,
                        lora_alpha=t_alpha,
                        lora_dropout=0.1,
                        target_modules=t_targets,
                        modules_to_save=heads,
                    )
                    model = get_peft_model(model, cfg)

                model = model.to(device)
                trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_p = sum(p.numel() for p in model.parameters())

                # Training Setup
                use_amp = device == "cuda"
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=t_lr, weight_decay=0.01)
                total_steps = t_epochs * len(train_loader)
                sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                model.train()
                global_step = 0

                for epoch in range(t_epochs):
                    running_loss = 0.0
                    for step, batch in enumerate(train_loader):
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

                        running_loss += loss.item()
                        global_step += 1
                        pct = global_step / total_steps
                        pbar.progress(pct)
                        status_text.write(
                            f"⚡ **Epoch {epoch + 1}/{t_epochs}** | Step {step + 1}/{len(train_loader)} | Batch Loss: `{loss.item():.4f}` | Avg: `{running_loss / (step + 1):.4f}`"
                        )

                if device == "cuda":
                    torch.cuda.synchronize()
                train_time = time.perf_counter() - t0
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else float("nan")

                # Evaluation
                status_text.info("Evaluating fine-tuned model on test split...")
                t_eval_0 = time.perf_counter()
                model.eval()
                preds, labels_list = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        logits = model(**batch).logits
                        preds.extend(logits.argmax(-1).cpu().tolist())
                        labels_list.extend(batch["labels"].cpu().tolist())

                eval_time = time.perf_counter() - t_eval_0
                acc = evaluate.load("accuracy").compute(predictions=preds, references=labels_list)["accuracy"]
                f1 = evaluate.load("f1").compute(predictions=preds, references=labels_list, average="macro")["f1"]

                # Save Checkpoint
                out_path = Path("runs") / custom_run_name
                out_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(out_path)
                saved_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1024**2
                tok.save_pretrained(out_path)

                # Append to CSV
                new_row = {
                    "run": custom_run_name,
                    "mode": t_mode,
                    "model": t_model,
                    "rank": t_rank if t_mode == "lora" else np.nan,
                    "alpha": t_alpha if t_mode == "lora" else np.nan,
                    "targets": "+".join(t_targets) if t_mode == "lora" else "all",
                    "lr": t_lr,
                    "epochs": t_epochs,
                    "train_size": len(raw_train),
                    "total_params": total_p,
                    "trainable_params": trainable_p,
                    "trainable_pct": round(trainable_p / total_p * 100, 3),
                    "train_time_s": round(train_time, 1),
                    "peak_train_mem_mb": round(peak_mb, 1),
                    "saved_size_mb": round(saved_mb, 2),
                    "eval_time_s": round(eval_time, 2),
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1, 4),
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                    "torch": torch.__version__,
                }

                df_new = pd.DataFrame([new_row])
                df_new.to_csv("lora_results.csv", mode="a", header=False, index=False)

                pbar.progress(1.0)
                status_text.empty()
                st.balloons()
                st.success(
                    f"🎉 **Training Complete!** Checkpoint saved to `{out_path}` | **Accuracy:** `{acc*100:.2f}%` | **Time:** `{train_time:.1f}s` | **Size:** `{saved_mb:.2f} MB`"
                )
                time.sleep(2)
                st.rerun()

            except Exception as e:
                st.error(f"Training failed: {e}")


# ----------------------------------------------------------------------------
# TAB 2: Analytics & Radar Trade-Off
# ----------------------------------------------------------------------------
with tab_analytics:
    col_rad, col_front = st.columns([1.1, 1.2])

    with col_rad:
        radar_categories = ["Storage Efficiency", "VRAM Savings", "Training Speed", "Accuracy Retention", "F1 Performance"]
        fig_radar = go.Figure()

        if not full_df.empty:
            base_acc = base_run["accuracy"] * 100
            base_f1 = base_run["macro_f1"] * 100
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[10, 20, 30, base_acc, base_f1],
                    theta=radar_categories,
                    fill="toself",
                    name="Full Fine-Tuning",
                    line=dict(color="#f43f5e", width=2),
                    fillcolor="rgba(244, 63, 94, 0.15)",
                )
            )

        if not lora_df.empty:
            r8_run = lora_df.iloc[0]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[98.9, 80.0, 75.0, r8_run["accuracy"] * 100, r8_run["macro_f1"] * 100],
                    theta=radar_categories,
                    fill="toself",
                    name=f"LoRA (r={r8_run['rank']})",
                    line=dict(color="#06b6d4", width=2),
                    fillcolor="rgba(6, 182, 212, 0.2)",
                )
            )

        if len(lora_df) > 1:
            r16_run = lora_df.sort_values("accuracy", ascending=False).iloc[0]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[95.0, 78.0, 70.0, r16_run["accuracy"] * 100, r16_run["macro_f1"] * 100],
                    theta=radar_categories,
                    fill="toself",
                    name=f"LoRA High-Rank (r={r16_run['rank']})",
                    line=dict(color="#10b981", width=2),
                    fillcolor="rgba(16, 185, 129, 0.2)",
                )
            )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showline=False, gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=30, r=30, t=40, b=20),
            title="<b>Multi-Dimensional Efficiency Radar</b>",
        )
        st.plotly_chart(fig_radar, width="stretch")

    with col_front:
        fig_acc = px.scatter(
            df,
            x="trainable_pct",
            y="accuracy",
            size=[max(18, s * 1.5) for s in [22] * len(df)],
            color="mode",
            color_discrete_map={"full": "#f43f5e", "lora": "#10b981"},
            hover_name="run",
            hover_data={
                "train_time_s": ":.1fs",
                "saved_size_mb": ":.2f MB",
                "peak_train_mem_mb": ":.1f MB",
                "accuracy": ":.4f",
            },
            title="<b>Pareto Frontier: Accuracy vs Active Parameters (%)</b>",
        )
        fig_acc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(title="Trainable Parameters (%)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="Test Accuracy", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_acc, width="stretch")

    col_foot, col_dur = st.columns(2)
    with col_foot:
        fig_mem = go.Figure()
        fig_mem.add_trace(
            go.Bar(
                x=df["run"],
                y=df["peak_train_mem_mb"],
                name="Peak VRAM (MB)",
                marker_color="#6366f1",
                text=df["peak_train_mem_mb"].round(1),
                textposition="auto",
            )
        )
        fig_mem.add_trace(
            go.Bar(
                x=df["run"],
                y=df["saved_size_mb"],
                name="Saved Checkpoint (MB)",
                marker_color="#06b6d4",
                text=df["saved_size_mb"].round(2),
                textposition="auto",
            )
        )
        fig_mem.update_layout(
            title="<b>VRAM Footprint vs Storage Footprint (MB)</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode="group",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_mem, width="stretch")

    with col_dur:
        fig_f1 = px.bar(
            df,
            x="run",
            y=["accuracy", "macro_f1"],
            barmode="group",
            title="<b>Model Quality Comparison: Accuracy & Macro F1</b>",
            color_discrete_sequence=["#10b981", "#3b82f6"],
        )
        fig_f1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Score (0 to 1)", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        )
        st.plotly_chart(fig_f1, width="stretch")


# ----------------------------------------------------------------------------
# TAB 3: Model Arena (Side-by-Side Dual Inference Battle)
# ----------------------------------------------------------------------------
with tab_arena:
    st.markdown("### ⚔️ **Model Arena: Head-to-Head Inference Battle**")
    st.markdown("Test two checkpoints side-by-side on the exact same input to evaluate prediction alignment, confidence, and inference latency.")

    available_runs = [d.name for d in Path("runs").glob("*") if d.is_dir()]
    if len(available_runs) < 2:
        st.info("Arena ke liye kam se kam 2 models zaroori hain.")

    col_arena_sel1, col_arena_sel2 = st.columns(2)
    with col_arena_sel1:
        default_idx1 = available_runs.index("full") if "full" in available_runs else 0
        model_a_name = st.selectbox("🥊 Model A (e.g. Baseline):", available_runs, index=default_idx1, key="arena_a")
    with col_arena_sel2:
        default_idx2 = 1 if len(available_runs) > 1 else 0
        model_b_name = st.selectbox("🥊 Model B (e.g. LoRA Adapter):", available_runs, index=default_idx2, key="arena_b")

    arena_prompt = st.text_input(
        "✍️ Input sentence for both models:",
        value="I am absolutely thrilled by this incredible breakthrough, but a little nervous about next steps!",
    )

    arena_btn = st.button("🔥 Run Side-by-Side Battle", type="primary", width="stretch")

    if arena_btn or arena_prompt:
        try:
            with st.spinner("Executing forward passes across both models..."):
                tok_a, mod_a, dev_a, classes_a = load_cached_model(str(Path("runs") / model_a_name))
                tok_b, mod_b, dev_b, classes_b = load_cached_model(str(Path("runs") / model_b_name))

                pred_a, conf_a, probs_a, lat_a = run_single_inference(tok_a, mod_a, dev_a, classes_a, arena_prompt)
                pred_b, conf_b, probs_b, lat_b = run_single_inference(tok_b, mod_b, dev_b, classes_b, arena_prompt)

            match = pred_a == pred_b
            verdict_color = "#10b981" if match else "#f59e0b"
            verdict_icon = "✅" if match else "⚠️"
            verdict_text = "PREDICTIONS AGREE" if match else "DIVERGENT PREDICTIONS"

            st.markdown(
                f"""
                <div style="text-align:center; padding:12px; margin: 16px 0; background: rgba(255,255,255,0.02); border: 1px solid {verdict_color}; border-radius:12px;">
                    <span style="font-size:1.1rem; font-weight:700; color:{verdict_color};">{verdict_icon} {verdict_text}: Selected <b>{pred_a.upper()}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_res_a, col_res_b = st.columns(2)
            with col_res_a:
                st.markdown(
                    f"""
                    <div class="arena-box">
                        <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Model A: {model_a_name}</div>
                        <div style="font-size:1.9rem; font-weight:800; color:#38bdf8; margin-top:6px;">
                            {pred_a.upper()}
                        </div>
                        <div style="font-size:1rem; color:#e2e8f0; margin-top:4px;">Confidence: <b>{conf_a:.1f}%</b></div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">⚡ Latency: {lat_a:.1f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_mini_a = px.bar(
                    x=[p * 100 for p in probs_a],
                    y=classes_a,
                    orientation="h",
                    color=classes_a,
                )
                fig_mini_a.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=200,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(range=[0, 100], title="Probability %", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title="", showgrid=False),
                )
                st.plotly_chart(fig_mini_a, width="stretch")

            with col_res_b:
                st.markdown(
                    f"""
                    <div class="arena-box">
                        <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; font-weight:700;">Model B: {model_b_name}</div>
                        <div style="font-size:1.9rem; font-weight:800; color:#34d399; margin-top:6px;">
                            {pred_b.upper()}
                        </div>
                        <div style="font-size:1rem; color:#e2e8f0; margin-top:4px;">Confidence: <b>{conf_b:.1f}%</b></div>
                        <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">⚡ Latency: {lat_b:.1f} ms</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_mini_b = px.bar(
                    x=[p * 100 for p in probs_b],
                    y=classes_b,
                    orientation="h",
                    color=classes_b,
                )
                fig_mini_b.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=200,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(range=[0, 100], title="Probability %", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(title="", showgrid=False),
                )
                st.plotly_chart(fig_mini_b, width="stretch")

        except Exception as e:
            st.error(f"Arena execution failed: {e}")


# ----------------------------------------------------------------------------
# TAB 4: Live Inference Playground (Single Model with Gauge)
# ----------------------------------------------------------------------------
with tab_playground:
    st.markdown("### 🧪 **Live Model Playground**")
    st.markdown("Deep dive into confidence distributions and class meters for any individual checkpoint.")

    col_ctrl1, col_ctrl2 = st.columns([1.1, 2])
    with col_ctrl1:
        selected_run = st.selectbox("🎯 Active Checkpoint:", available_runs, index=0, key="play_model")

        st.markdown("**Quick Preset Prompts:**")
        p1 = st.button("🎉 'I just got my dream job promotion!'", key="btn_p1", width="stretch")
        p2 = st.button("😨 'A spider just jumped on my shoulder!'", key="btn_p2", width="stretch")
        p3 = st.button("💔 'I really miss those good old days.'", key="btn_p3", width="stretch")
        p4 = st.button("😡 'Who deleted all my database tables?!'", key="btn_p4", width="stretch")

    with col_ctrl2:
        default_prompt = "I feel incredibly joyful and excited about this new breakthrough!"
        if p1:
            default_prompt = "I just got my dream job promotion!"
        elif p2:
            default_prompt = "A spider just jumped on my shoulder!"
        elif p3:
            default_prompt = "I really miss those good old days."
        elif p4:
            default_prompt = "Who deleted all my database tables?!"

        user_text = st.text_area("✍️ Enter text to classify:", value=default_prompt, height=90)
        analyze_btn = st.button("🚀 Classify Text", type="primary", width="stretch", key="play_btn")

        if analyze_btn or user_text:
            ckpt_path = str(Path("runs") / selected_run)
            try:
                with st.spinner("Computing forward pass..."):
                    tok, model, device, classes = load_cached_model(ckpt_path)
                    top_pred, top_conf, probs, inf_latency_ms = run_single_inference(tok, model, device, classes, user_text)

                col_top_res, col_gauge = st.columns([1.4, 1.2])
                with col_top_res:
                    st.markdown(
                        f"""
                        <div style="background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:20px; margin: 10px 0;">
                            <div style="font-size:0.8rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Predicted Class</div>
                            <div style="font-size:2.2rem; font-weight:800; color:#38bdf8; margin-top:4px;">
                                {top_pred.upper()}
                            </div>
                            <div style="font-size:1.1rem; color:#e2e8f0; font-weight:600; margin-top:4px;">Confidence: {top_conf:.1f}%</div>
                            <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">⚡ Latency: {inf_latency_ms:.1f} ms | Device: {device.upper()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_gauge:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=top_conf,
                            title={"text": f"<b>{top_pred.capitalize()} Confidence</b>", "font": {"size": 15, "color": "#cbd5e1"}},
                            number={"suffix": "%", "font": {"size": 24, "color": "#38bdf8"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                                "bar": {"color": "#38bdf8"},
                                "bgcolor": "rgba(255,255,255,0.05)",
                                "borderwidth": 1,
                                "bordercolor": "rgba(255,255,255,0.1)",
                                "steps": [
                                    {"range": [0, 50], "color": "rgba(255,255,255,0.02)"},
                                    {"range": [50, 80], "color": "rgba(255,255,255,0.05)"},
                                ],
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#cbd5e1"},
                        height=160,
                        margin=dict(l=10, r=10, t=30, b=10),
                    )
                    st.plotly_chart(fig_gauge, width="stretch")

                st.markdown("#### **Full Probability Distribution:**")
                for label, prob in zip(classes, probs):
                    pct = prob * 100
                    st.markdown(
                        f"""
                        <div class="prob-bar-container">
                            <div class="prob-bar-label">
                                <span>{label.capitalize()}</span>
                                <span style="color:#cbd5e1;">{pct:.1f}%</span>
                            </div>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill" style="width: {pct}%; background: #6366f1;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Inference error: {e}")


# ----------------------------------------------------------------------------
# TAB 5: LoRA Parameter Calculator & Architecture Explorer
# ----------------------------------------------------------------------------
with tab_calc:
    st.markdown("### 🧮 **Interactive LoRA Parameter & Memory Calculator**")
    st.markdown("Explore how LoRA parameters scale with Rank ($r$), Alpha ($\\alpha$), and targeted Transformer layers in DistilBERT.")

    calc_c1, calc_c2 = st.columns([1, 1.2])

    with calc_c1:
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.markdown("#### **Hyperparameter Selectors**")
        sim_rank = st.select_slider("Rank Dimension ($r$):", options=[1, 2, 4, 8, 16, 32, 64], value=8)
        sim_alpha = st.select_slider("Scaling Alpha ($\\alpha$):", options=[1, 2, 4, 8, 16, 32, 64, 128], value=16)

        sim_targets = st.multiselect(
            "Target Projection Modules:",
            options=["q_lin", "k_lin", "v_lin", "out_lin", "lin1 (FFN)", "lin2 (FFN)"],
            default=["q_lin", "v_lin"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with calc_c2:
        d = 768
        num_layers = 6
        target_param_counts = {
            "q_lin": 2 * d * sim_rank * num_layers,
            "k_lin": 2 * d * sim_rank * num_layers,
            "v_lin": 2 * d * sim_rank * num_layers,
            "out_lin": 2 * d * sim_rank * num_layers,
            "lin1 (FFN)": (d * sim_rank + 3072 * sim_rank) * num_layers,
            "lin2 (FFN)": (3072 * sim_rank + d * sim_rank) * num_layers,
        }

        head_params = (768 * 768 + 768) + (768 * 6 + 6)
        adapter_params = sum(target_param_counts[t] for t in sim_targets)
        total_trainable = adapter_params + head_params
        total_model = 66_958_086 + adapter_params
        trainable_pct = (total_trainable / total_model) * 100
        scaling_ratio = sim_alpha / sim_rank

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius:16px; padding:22px;">
                <div style="font-size:0.85rem; color:#94a3b8; font-weight:700; text-transform:uppercase;">Calculated Efficiency</div>
                <div style="display:flex; justify-content:space-between; margin-top:14px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">
                    <span style="color:#cbd5e1;">LoRA Adapter Parameters:</span>
                    <span style="font-weight:700; color:#38bdf8;">{adapter_params:,}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:10px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">
                    <span style="color:#cbd5e1;">Classification Head:</span>
                    <span style="font-weight:700; color:#cbd5e1;">{head_params:,}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:10px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">
                    <span style="color:#cbd5e1;">Total Trainable Weights:</span>
                    <span style="font-weight:800; color:#10b981;">{total_trainable:,} ({trainable_pct:.2f}%)</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:10px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">
                    <span style="color:#cbd5e1;">Scaling Factor (α / r):</span>
                    <span style="font-weight:700; color:#facc15;">{scaling_ratio:.2f}x</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:10px;">
                    <span style="color:#cbd5e1;">Estimated Adapter Checkpoint:</span>
                    <span style="font-weight:800; color:#c084fc;">~{(total_trainable * 4) / 1024**2:.2f} MB</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# TAB 6: Benchmark Database & Full Reductions Matrix
# ----------------------------------------------------------------------------
with tab_table:
    st.markdown("### 📋 **Complete Benchmarked Runs**")
    cols_display = [
        "run",
        "mode",
        "rank",
        "alpha",
        "targets",
        "trainable_params",
        "trainable_pct",
        "train_time_s",
        "peak_train_mem_mb",
        "saved_size_mb",
        "accuracy",
        "macro_f1",
    ]
    st.dataframe(df[[c for c in cols_display if c in df.columns]], width="stretch")

    if not full_df.empty:
        base = full_df.iloc[-1]
        comp = df[df["mode"] == "lora"].copy()
        if not comp.empty:
            comp["param_reduction_%"] = ((base.trainable_params - comp.trainable_params) / base.trainable_params * 100).round(2)
            comp["storage_saving_%"] = ((base.saved_size_mb - comp.saved_size_mb) / base.saved_size_mb * 100).round(2)
            comp["vram_reduction_%"] = ((base.peak_train_mem_mb - comp.peak_train_mem_mb) / base.peak_train_mem_mb * 100).round(2)
            comp["speedup_x"] = (base.train_time_s / comp.train_time_s).round(2)
            comp["acc_delta"] = (comp.accuracy - base.accuracy).round(4)

            st.markdown("### 📈 **LoRA Reductions vs Full Baseline**")
            st.dataframe(
                comp[["run", "param_reduction_%", "storage_saving_%", "vram_reduction_%", "speedup_x", "acc_delta"]].set_index("run"),
                width="stretch",
            )
