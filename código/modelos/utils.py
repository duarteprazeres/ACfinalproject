"""
utils.py — Utilitários partilhados entre model_svm.py, model_nn.py e model_xgboost.py
======================================================================================
Contém: métricas, logging CSV, gráficos e split de dados.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, classification_report
)

# =============================================================
# CONSTANTES GLOBAIS
# =============================================================
RESULTS_CSV  = 'experiment_results.csv'
MODELS_DIR   = Path('models')
GRAFICOS_DIR = Path('graficos')
RANDOM_STATE = 42
CV_FOLDS     = 5
TEST_SIZE    = 0.15
VAL_RATIO    = 0.15 / 0.85

MODELS_DIR.mkdir(exist_ok=True)
GRAFICOS_DIR.mkdir(exist_ok=True)


# =============================================================
# 2.2 MÉTRICAS
# =============================================================
def compute_metrics(y_true, y_pred, y_proba=None, scenario='A'):
    """
    Calcula métricas adequadas a cada cenário:
      A (binário):  F1-macro, Precision, Recall, ROC-AUC, MCC
      B (5 cls):    F1-macro, Weighted-F1, Cohen's Kappa
      C (10 cls):   F1-macro, Weighted-F1, Cohen's Kappa, Top-2 Accuracy
    """
    metrics = {
        'f1_macro':      f1_score(y_true, y_pred, average='macro',    zero_division=0),
        'f1_weighted':   f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_mac': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_mac':    recall_score(y_true, y_pred, average='macro',    zero_division=0),
        'cohen_kappa':   cohen_kappa_score(y_true, y_pred),
        'mcc':           matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            if scenario == 'A':
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
        except Exception:
            metrics['roc_auc'] = float('nan')

    if scenario == 'C' and y_proba is not None:
        try:
            top2 = np.argsort(y_proba, axis=1)[:, -2:]
            y_arr = np.array(y_true)
            metrics['top2_accuracy'] = float(
                np.mean([y_arr[i] in top2[i] for i in range(len(y_arr))])
            )
        except Exception:
            metrics['top2_accuracy'] = float('nan')

    return metrics


# =============================================================
# 2.8 LOGGING CSV
# =============================================================
def log_experiment(model_name, scenario, params, metrics, split='val'):
    """Acrescenta uma linha ao experiment_results.csv."""
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model':     model_name,
        'scenario':  scenario,
        'split':     split,
        'params':    json.dumps(params),
    }
    row.update({k: round(float(v), 6) if not isinstance(v, str) else v
                for k, v in metrics.items()})
    file_exists = os.path.isfile(RESULTS_CSV)
    pd.DataFrame([row]).to_csv(
        RESULTS_CSV, mode='a', header=not file_exists, index=False
    )


# =============================================================
# GRÁFICOS
# =============================================================
def plot_confusion_matrix_fig(y_true, y_pred, model_name, scenario_name):
    """Confusion matrix normalizada → graficos/."""
    cm  = confusion_matrix(y_true, y_pred, normalize='true')
    n   = len(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='.2f')
    ax.set_title(f'Confusion Matrix (Norm.) — {model_name} | Cenário {scenario_name}')
    plt.tight_layout()
    fname = GRAFICOS_DIR / f'cm_{model_name.lower().replace(" ", "_")}_{scenario_name}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"   -> Confusion matrix: {fname}")


def plot_cv_baseline(cv_rows, model_name):
    """Gráfico de barras com F1 mean ± std do CV baseline."""
    df = pd.DataFrame(cv_rows)
    palette = {'A': '#4C72B0', 'B': '#DD8452', 'C': '#55A868'}
    colors  = [palette.get(r['scenario'], '#888') for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(df)), df['f1_mean'], yerr=df['f1_std'],
                  capsize=5, color=colors, alpha=0.87)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"Cenário {r['scenario']}" for _, r in df.iterrows()])
    ax.set_ylabel('F1 Macro (mean ± std)')
    ax.set_title(f'Cross-Validation Baseline — {model_name}')
    ax.set_ylim(0, 1.05)
    for bar, m, s in zip(bars, df['f1_mean'], df['f1_std']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{m:.3f}±{s:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    tag  = model_name.lower().replace('/', '_').replace(' ', '_')
    path = GRAFICOS_DIR / f'cv_baseline_{tag}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   -> CV baseline chart: {path}")


# =============================================================
# DIVISÃO DOS DADOS
# =============================================================
def split_scenario(X_raw, y, scenario_name):
    """Split estratificado 70/15/15. Devolve arrays sem normalização."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_RATIO, stratify=y_tmp, random_state=RANDOM_STATE
    )
    print(f"   [{scenario_name}] Train:{len(X_train)} | Val:{len(X_val)} | Test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================
# PERSISTÊNCIA
# =============================================================
def save_model(obj, filename):
    """Guarda um objecto com joblib."""
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    print(f"   Guardado: {path}")
    return path


def update_registry(registry_updates: dict):
    """Acrescenta/actualiza entradas no model_registry.json."""
    reg_path = MODELS_DIR / 'model_registry.json'
    if reg_path.exists():
        with open(reg_path) as f:
            registry = json.load(f)
    else:
        registry = {}
    registry.update(registry_updates)
    with open(reg_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)
    print(f"   Registry actualizado: {reg_path}")
