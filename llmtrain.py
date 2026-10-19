#!/usr/bin/env python
"""Standalone Cell 5: compare gold-only with gold + pseudo-label training.

Run Cell 4 successfully once to create combined_train.csv and dataset_audit.json.
This script does not need a running notebook or import cell_4.py.

Example (terminal):
    python train_cell5.py --project-dir "C:/path/to/notebook-folder"
Validate the saved datasets without loading torch or downloading a model:
    python train_cell5.py --project-dir "C:/path/to/notebook-folder" --check-only

Dependencies: torch, transformers, datasets, accelerate, numpy, pandas,
scikit-learn, scipy, and tokenizer dependencies (e.g. sentencepiece).
Use your existing working PyTorch environment. A separate script cannot repair
a broken PyTorch DLL installation.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_dataframe(path):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False, dtype={'text': 'string'})
    missing = {'text', 'label'} - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    df = df[['text', 'label']].copy()
    labels = pd.to_numeric(df['label'], errors='raise')
    if not labels.isin([0, 1]).all():
        raise ValueError(f'{path}: labels must be exactly 0 or 1 (no missing/fractional values)')
    if df.empty or df['text'].isna().any() or df['text'].str.strip().eq('').any():
        raise ValueError(f'{path}: empty dataset or missing/blank text; repair the source split')
    df['text'] = df['text'].astype(str)
    df['label'] = labels.astype('int64')
    return df.reset_index(drop=True)

def text_keys(df):
    return df['text'].str.strip()

def check_disjoint(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(text_keys(splits[a])) & set(text_keys(splits[b]))
            if overlap:
                raise ValueError(f'{a}/{b}: {len(overlap):,} overlapping texts; repair splits before training')

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_inputs(project_dir):
    proc_base = project_dir / '.'
    res_base = project_dir / '.'
    gold_files = {s: proc_base / f'gold_{s}.csv' for s in ('train', 'val', 'test')}
    combined_file = proc_base / 'combined_train.csv'
    audit_path = res_base / 'dataset_audit.json'
    required = [*gold_files.values(), combined_file, audit_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n' + '\n'.join(missing) +
                                '\nRun Cell 4 successfully and use its working folder as --project-dir.')
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit.get('ai_rows') != 101_000:
        raise ValueError('dataset_audit.json must record exactly 101,000 AI annotation rows.')
    hashes = audit.get('files')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Dataset audit has no file hashes. Rerun the revised Cell 4.')
    verified = set()
    for recorded_path, expected in hashes.items():
        path = Path(recorded_path)
        if not path.is_absolute():
            path = project_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Audited input missing: {path}. Keep the Cell 4 inputs available.')
        if file_sha256(path) != expected:
            raise ValueError(f'Input changed since Cell 4: {path}. Rerun Cell 4.')
        verified.add(path)
    if not {p.resolve() for p in [*gold_files.values(), combined_file]}.issubset(verified):
        raise ValueError('The audit does not cover these training/evaluation files. Rerun Cell 4.')

    frames = {s: load_dataframe(p) for s, p in gold_files.items()}
    frames['combined'] = load_dataframe(combined_file)
    check_disjoint({s: frames[s] for s in ('train', 'val', 'test')})
    check_disjoint({s: frames[s] for s in ('combined', 'val', 'test')})
    for split, df in frames.items():
        if set(df['label']) != {0, 1}:
            raise ValueError(f'{split} must contain both classes for this comparison.')
    if len(frames['combined']) != audit.get('combined_train_rows'):
        raise ValueError('Combined row count does not match Cell 4 audit.')
    print('Verified Cell 4 audit: 101,000 AI annotation input rows.')
    print('Dataset sizes:', {s: len(df) for s, df in frames.items()})
    print('Both experiments use the same human-gold validation and test sets.')
    return frames, audit, res_base


def run_training(frames, audit, RES_BASE, project_dir):
    import torch
    import os
    import gc
    import random
    import inspect
    from datetime import datetime
    import uuid
    import numpy as np
    import pandas as pd
    from datasets import Dataset as HFDataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, EarlyStoppingCallback)
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, matthews_corrcoef, roc_auc_score,
                                 average_precision_score)
    from scipy import stats


    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    SAHAJ = 'neuropark/sahajBERT'  # Base model, not the 191k-corpus DAPT checkpoint
    SEEDS_5 = [42, 123, 999, 2024, 7]
    MAX_LENGTH = 128
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    USE_FP16 = torch.cuda.is_available()
    RUN_DIR = RES_BASE / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    print('Device:', DEVICE, '| Results:', RUN_DIR)

    # ── Reproducibility: same style as the master setup ──────────
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PYTHONHASHSEED must be set before starting Python to affect this process.

    # ── Metrics: weighted scores match annotation/3seeds ─────────
    # Keep full precision for checkpoint selection and statistical comparisons.
    def compute_all_metrics(y_true, y_pred, y_prob=None):
        m = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'f1_depressive': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        }
        if y_prob is not None:
            m['auc'] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
            m['average_precision'] = (average_precision_score(y_true, y_prob)
                                      if len(np.unique(y_true)) == 2 else np.nan)
        return m

    def probabilities(logits):
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits, dtype=np.float64)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        prob = probabilities(logits)
        return compute_all_metrics(labels, prob.argmax(axis=1), prob[:, 1])

    tokenizer = AutoTokenizer.from_pretrained(SAHAJ)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True,
                         padding='max_length', max_length=MAX_LENGTH)

    def build_dataset(df):
        ds = HFDataset.from_pandas(df, preserve_index=False)
        ds = ds.map(tokenize, batched=True, remove_columns=['text'])
        cols = [c for c in tokenizer.model_input_names if c in ds.column_names]
        ds.set_format(type='torch', columns=cols + ['label'])
        return ds

    datasets = {s: build_dataset(df) for s, df in frames.items()}
    arg_names = inspect.signature(TrainingArguments.__init__).parameters
    EVAL_KEY = 'eval_strategy' if 'eval_strategy' in arg_names else 'evaluation_strategy'
    all_results = []

    def run_experiment(train_key, experiment_name, seeds):
        results = []
        for seed in seeds:
            set_seed(seed)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Use the normal model loader. Do not disable torch-load safety checks.
            model = AutoModelForSequenceClassification.from_pretrained(
                SAHAJ, num_labels=2, ignore_mismatched_sizes=True)
            output_dir = RUN_DIR / f'{experiment_name}_{seed}'
            args = TrainingArguments(
                output_dir=str(output_dir),
                **{EVAL_KEY: 'epoch'},
                save_strategy='epoch', logging_strategy='epoch', save_total_limit=1,
                learning_rate=2e-5,
                per_device_train_batch_size=32, per_device_eval_batch_size=64,
                num_train_epochs=8, warmup_steps=0.1, weight_decay=0.01,
                fp16=USE_FP16, dataloader_num_workers=0,
                seed=seed, data_seed=seed, report_to='none',
                load_best_model_at_end=True, metric_for_best_model='f1',
                greater_is_better=True,
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=datasets[train_key],
                eval_dataset=datasets['val'], compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )
            try:
                trainer.train()
                prediction = trainer.predict(datasets['test'])
                prob = probabilities(prediction.predictions)
                m = compute_all_metrics(prediction.label_ids, prob.argmax(axis=1), prob[:, 1])
                m.update(experiment=experiment_name, seed=seed,
                         train_rows=len(frames[train_key]),
                         best_checkpoint=trainer.state.best_model_checkpoint)
                results.append(m)
                all_results.append(m)
                pd.DataFrame(all_results).to_csv(RUN_DIR / 'semisupervised_comparison.csv', index=False)
                pd.DataFrame({
                    'test_row': np.arange(len(prediction.label_ids)),
                    'label': prediction.label_ids, 'prediction': prob.argmax(axis=1),
                    'prob_depressive': prob[:, 1],
                }).to_csv(RUN_DIR / f'predictions_{experiment_name}_{seed}.csv', index=False)
                print(f"{experiment_name} seed={seed}: weighted F1={m['f1']:.4f}, MCC={m['mcc']:.4f}")
            finally:
                del trainer, model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return results

    # Save settings and input hashes alongside results; preserve best checkpoints.
    import transformers
    config = {'model': SAHAJ, 'seeds': SEEDS_5, 'max_length': MAX_LENGTH,
              'selection_metric': 'weighted F1', 'torch': torch.__version__,
              'transformers': transformers.__version__, 'dataset_audit': audit,
              'project_dir': str(project_dir), 'python': sys.version,
              'python_executable': sys.executable}
    (RUN_DIR / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    results_gold = run_experiment('train', 'gold_only', SEEDS_5)
    results_combined = run_experiment('combined', 'combined', SEEDS_5)

    # ── Matched-seed comparison ─────────────────────────────────
    df_res = pd.DataFrame(all_results)
    paired = df_res.pivot(index='seed', columns='experiment', values='f1').loc[SEEDS_5]
    delta = paired['combined'] - paired['gold_only']
    summary = df_res.groupby('experiment')[['f1', 'mcc', 'accuracy', 'auc']].agg(['mean', 'std'])
    summary.to_csv(RUN_DIR / 'summary.csv')
    print('\nFINAL COMPARISON — weighted F1, sample SD')
    for name in ('gold_only', 'combined'):
        print(f'{name}: {paired[name].mean():.4f} ± {paired[name].std(ddof=1):.4f}')
    print(f'Mean difference: {delta.mean():+.4f}')
    if np.allclose(delta, delta.iloc[0], rtol=0, atol=1e-12):
        print('Paired t-test not reported: seed differences have zero variance.')
    else:
        t, p = stats.ttest_rel(paired['combined'], paired['gold_only'])
        print(f'Exploratory paired t-test: t={t:.4f}, p={p:.4g}')
    print('Five seed pairs describe training variability on one fixed test split; '
          'they do not quantify uncertainty across new test samples.')
    print('Report all seeds and the direction/magnitude of the change; '
          'do not infer that pseudo-labeling helps from the mean alone.')
    print('Saved results:', RUN_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Working folder used by Cell 4; defaults to this script folder.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate saved datasets/audit without loading PyTorch or training.')
    args = parser.parse_args()
    project_dir = args.project_dir.expanduser().resolve()
    print('Python:', sys.executable)
    print('Project folder:', project_dir)
    if not args.check_only:
        # First third-party import in the standalone process.
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                'PyTorch could not load in a fresh process. The DLL/package issue '
                'must be repaired in this Python environment before training. '
                f'Python executable: {sys.executable}\nOriginal error: {exc}'
            ) from exc
    frames, audit, res_base = validate_inputs(project_dir)
    if args.check_only:
        print('Input checks passed. No model was loaded and no training was performed.')
        return
    run_training(frames, audit, res_base, project_dir)


if __name__ == '__main__':
    main()
