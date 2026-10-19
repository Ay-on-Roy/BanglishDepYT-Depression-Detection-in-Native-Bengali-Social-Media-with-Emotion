# run_all_models.py
# Complete unified experiment: all 4 transformer models
# Same hyperparameters, same gold dataset, same 10 seeds
# Run from: F:\user1\deps\
# Output: results/FINAL_UNIFIED_RESULTS.csv

import os
import gc
import torch
import random
import numpy as np
import pandas as pd
from scipy import stats
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    matthews_corrcoef
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import transformers.modeling_utils as mu
mu.check_torch_load_is_safe = lambda: None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ──────────────────────────────────────────────────
BASE        = r"F:\user1\deps"
GOLD_TRAIN  = os.path.join(BASE, "gold_train.csv")
GOLD_VAL    = os.path.join(BASE, "gold_val.csv")
GOLD_TEST   = os.path.join(BASE, "gold_test.csv")
DAPT_CKPT   = os.path.join(BASE, "dapt_sahajbert")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ─────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Seeds ──────────────────────────────────────────────────
SEEDS_10 = [42, 123, 999, 2024, 7, 88, 314, 777, 1001, 5555]

# ── Models to run ──────────────────────────────────────────
MODELS = {
    "SahajBERT_baseline": "neuropark/sahajBERT",
    "DAPT_SahajBERT":     DAPT_CKPT,
    "MuRIL":              "google/muril-base-cased",
    "IndicBERTv2":        "ai4bharat/indic-bert",
}

# ── Hyperparameters (identical for all four models) ────────
LR           = 2e-5
BATCH_TRAIN  = 32
BATCH_EVAL   = 64
EPOCHS       = 8
PATIENCE     = 3
WARMUP_STEPS = 0.1
WEIGHT_DECAY = 0.01
MAX_LEN      = 128


# ── Reproducibility ────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Metrics ────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_dep":    f1_score(labels, preds, pos_label=1, zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall":    recall_score(labels, preds, average="weighted", zero_division=0),
        "mcc":       matthews_corrcoef(labels, preds),
    }


# ── Load and verify gold splits ────────────────────────────
print("\nLoading gold splits...")
for path, name in [(GOLD_TRAIN,"train"),(GOLD_VAL,"val"),(GOLD_TEST,"test")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    print(f"  {name}: {len(df)} rows | labels: {df['label'].value_counts().to_dict()}")

# Verify DAPT checkpoint
if not os.path.exists(DAPT_CKPT):
    print(f"\nWARNING: DAPT checkpoint not found at {DAPT_CKPT}")
    print("Removing DAPT_SahajBERT from run list")
    del MODELS["DAPT_SahajBERT"]
else:
    print(f"\nDApt checkpoint found: {DAPT_CKPT}")


# ── Main training function ─────────────────────────────────
def run_model(model_name, model_id, seeds):
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"ID:    {model_id}")
    print(f"{'='*60}")

    results  = []
    all_preds = {}

    # Tokenize once per model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    train_df = pd.read_csv(GOLD_TRAIN)
    val_df   = pd.read_csv(GOLD_VAL)
    test_df  = pd.read_csv(GOLD_TEST)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_ds   = Dataset.from_pandas(val_df).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    for ds in [train_ds, val_ds, test_ds]:
        ds.set_format("torch", columns=cols)

    for seed in seeds:
        print(f"\n  Seed {seed}...")
        set_seed(seed)
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=2,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)

        out_dir = os.path.join(BASE, f"tmp_{model_name}_{seed}")

        args = TrainingArguments(
            output_dir                  = out_dir,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            save_total_limit            = 1,
            logging_strategy            = "epoch",
            learning_rate               = LR,
            per_device_train_batch_size = BATCH_TRAIN,
            per_device_eval_batch_size  = BATCH_EVAL,
            num_train_epochs            = EPOCHS,
            warmup_steps                = WARMUP_STEPS,
            weight_decay                = WEIGHT_DECAY,
            fp16                        = torch.cuda.is_available(),
            dataloader_num_workers      = 0,
            seed                        = seed,
            report_to                   = "none",
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1",
            greater_is_better           = True,
        )

        trainer = Trainer(
            model           = model,
            args            = args,
            train_dataset   = train_ds,
            eval_dataset    = val_ds,
            compute_metrics = compute_metrics,
            callbacks       = [EarlyStoppingCallback(
                                  early_stopping_patience=PATIENCE)],
        )

        trainer.train()

        # Evaluate on test set
        pred_out = trainer.predict(test_ds)
        preds    = np.argmax(pred_out.predictions, axis=1)
        true     = test_df["label"].values
        m        = compute_metrics((pred_out.predictions, true))

        row = {
            "model":     model_name,
            "seed":      seed,
            "accuracy":  round(m["accuracy"],  4),
            "f1":        round(m["f1"],        4),
            "f1_dep":    round(m["f1_dep"],    4),
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
            "mcc":       round(m["mcc"],       4),
        }
        results.append(row)
        all_preds[seed] = {"pred": preds, "true": true}

        print(f"  seed={seed} F1={m['f1']:.4f} "
              f"MCC={m['mcc']:.4f} F1_dep={m['f1_dep']:.4f}")

        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()

        # Clean up tmp folder
        import shutil
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

    del tokenizer
    return results, all_preds


# ── Run all models ─────────────────────────────────────────
ALL_RESULTS  = []
ALL_PREDS    = {}

for model_name, model_id in MODELS.items():
    results, preds = run_model(model_name, model_id, SEEDS_10)
    ALL_RESULTS.extend(results)
    ALL_PREDS[model_name] = preds

    # Save after each model in case of crash
    pd.DataFrame(ALL_RESULTS).to_csv(
        os.path.join(RESULTS_DIR, "FINAL_UNIFIED_RESULTS.csv"),
        index=False)
    print(f"\nCheckpoint saved after {model_name}")


# ── Aggregate statistics ───────────────────────────────────
df = pd.DataFrame(ALL_RESULTS)
df.to_csv(os.path.join(RESULTS_DIR, "FINAL_UNIFIED_RESULTS.csv"), index=False)

print(f"\n{'='*60}")
print("AGGREGATED RESULTS")
print(f"{'='*60}")

agg = df.groupby("model").agg(
    f1_mean     = ("f1",  "mean"),
    f1_std      = ("f1",  "std"),
    mcc_mean    = ("mcc", "mean"),
    mcc_std     = ("mcc", "std"),
    acc_mean    = ("accuracy", "mean"),
    f1dep_mean  = ("f1_dep",   "mean"),
    f1dep_std   = ("f1_dep",   "std"),
).round(4)

print(agg.to_string())

agg.to_csv(
    os.path.join(RESULTS_DIR, "FINAL_UNIFIED_AGG.csv"))


# ── Statistical significance ───────────────────────────────
print(f"\n{'='*60}")
print("STATISTICAL SIGNIFICANCE (paired t-test)")
print(f"{'='*60}")

models_present = df["model"].unique().tolist()

def get_f1s(model_name):
    return df[df["model"]==model_name]["f1"].values

def compare(name_a, name_b):
    if name_a not in models_present or name_b not in models_present:
        print(f"  Skipping {name_a} vs {name_b} — one not present")
        return
    a = get_f1s(name_a)
    b = get_f1s(name_b)
    t, p = stats.ttest_rel(b, a)
    d    = (b-a).mean() / ((b-a).std(ddof=1) + 1e-9)
    sig  = "✅ p<0.05" if p < 0.05 else f"p={p:.3f}"
    print(f"\n  {name_a} vs {name_b}:")
    print(f"    Δ F1 = {b.mean()-a.mean():+.4f}")
    print(f"    t={t:.3f}  p={p:.4f}  d={d:.3f}  {sig}")

compare("SahajBERT_baseline", "DAPT_SahajBERT")
compare("DAPT_SahajBERT",     "MuRIL")
compare("DAPT_SahajBERT",     "IndicBERTv2")
compare("MuRIL",              "IndicBERTv2")


# ── McNemar test: DAPT vs Baseline ────────────────────────
if ("SahajBERT_baseline" in ALL_PREDS and
    "DAPT_SahajBERT"     in ALL_PREDS):

    print(f"\n{'='*60}")
    print("McNEMAR TEST — DAPT vs Baseline (seed 42)")
    print(f"{'='*60}")

    from statsmodels.stats.contingency_tables import mcnemar

    seed    = 42
    true    = ALL_PREDS["SahajBERT_baseline"][seed]["true"]
    pred_b  = ALL_PREDS["SahajBERT_baseline"][seed]["pred"]
    pred_d  = ALL_PREDS["DAPT_SahajBERT"][seed]["pred"]

    both_c  = int(((pred_b==true) & (pred_d==true)).sum())
    b_only  = int(((pred_b==true) & (pred_d!=true)).sum())
    d_only  = int(((pred_b!=true) & (pred_d==true)).sum())
    both_w  = int(((pred_b!=true) & (pred_d!=true)).sum())

    table  = [[both_c, b_only], [d_only, both_w]]
    result = mcnemar(table, exact=False, correction=True)

    print(f"  Both correct:          {both_c}")
    print(f"  Baseline only correct: {b_only}")
    print(f"  DAPT only correct:     {d_only}")
    print(f"  Both wrong:            {both_w}")
    print(f"  McNemar p-value:       {result.pvalue:.4f}")
    if d_only > 0:
        print(f"  DAPT/baseline ratio:   {d_only/b_only:.1f}x")


# ── Per-seed table ─────────────────────────────────────────
print(f"\n{'='*60}")
print("PER-SEED F1 TABLE")
print(f"{'='*60}")
pivot = df.pivot(index="seed", columns="model", values="f1")
print(pivot.round(4).to_string())
pivot.to_csv(os.path.join(RESULTS_DIR, "PERSEED_F1_TABLE.csv"))


print(f"\n{'='*60}")
print("ALL DONE")
print(f"Results saved to: {RESULTS_DIR}")
print(f"  FINAL_UNIFIED_RESULTS.csv")
print(f"  FINAL_UNIFIED_AGG.csv")
print(f"  PERSEED_F1_TABLE.csv")
print(f"{'='*60}")