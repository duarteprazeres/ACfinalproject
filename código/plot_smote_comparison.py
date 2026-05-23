"""
Gera o gráfico comparativo de distribuição de classes no Cenário B
antes e depois do SMOTE +30%, guardando em graficos/smote_comparison_B.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Erro: imbalanced-learn não instalado.")
    sys.exit(1)

from código.data_pipeline import load_and_inspect_data, generate_targets, feature_engineering

RANDOM_STATE = 42
TEST_SIZE    = 0.15
VAL_RATIO    = 0.15 / 0.85
SMOTE_PCT    = 30


def main():
    os.makedirs('graficos', exist_ok=True)

    filepath = 'dataset/sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        filepath = 'sleep_mobile_stress_dataset_15000.csv'

    df      = load_and_inspect_data(filepath)
    df      = generate_targets(df)
    X_raw, df_fe = feature_engineering(df)

    valid = ~df_fe[['target_A', 'target_B_ew', 'target_C_ew']].isnull().any(axis=1)
    X_raw = X_raw[valid].reset_index(drop=True)
    df_fe = df_fe[valid].reset_index(drop=True)

    y_B = df_fe['target_B_ew'].astype(int)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_raw.values, y_B, test_size=TEST_SIZE, stratify=y_B, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_RATIO, stratify=y_tmp, random_state=RANDOM_STATE
    )

    counts_before = pd.Series(y_train).value_counts().sort_index()

    majority_count = int(counts_before.max())
    sampling_strategy = {}
    for cls, count in counts_before.items():
        if count >= majority_count:
            continue
        target = int(count * (1 + SMOTE_PCT / 100.0))
        if target > count:
            sampling_strategy[cls] = target

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    _, y_res = smote.fit_resample(X_train, y_train)
    counts_after = pd.Series(y_res).value_counts().sort_index()

    print("Antes SMOTE:", counts_before.to_dict())
    print("Depois SMOTE:", counts_after.to_dict())

    palette = sns.color_palette('viridis', 5)
    classes = [0, 1, 2, 3, 4]
    x = np.arange(len(classes))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # --- Antes ---
    bars_before = axes[0].bar(
        x, [counts_before.get(c, 0) for c in classes],
        color=palette, width=0.6, edgecolor='white', linewidth=0.8
    )
    axes[0].set_title('Antes do SMOTE', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Classe (Cenário B)', fontsize=11)
    axes[0].set_ylabel('Número de amostras (treino)', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Classe {c}' for c in classes])
    total_before = counts_before.sum()
    total_after  = counts_after.sum()
    ylim_top = max(counts_before.max(), counts_after.max()) * 1.20

    axes[0].set_ylim(0, ylim_top)
    axes[0].yaxis.grid(True, linestyle='--', linewidth=0.6, color='grey', alpha=0.5)
    axes[0].yaxis.set_minor_locator(plt.MultipleLocator(100))
    axes[0].yaxis.grid(True, which='minor', linestyle=':', linewidth=0.4, color='grey', alpha=0.3)
    axes[0].set_axisbelow(True)
    for bar, val in zip(bars_before, [counts_before.get(c, 0) for c in classes]):
        pct = val / total_before * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                     f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    # --- Depois ---
    bars_after = axes[1].bar(
        x, [counts_after.get(c, 0) for c in classes],
        color=palette, width=0.6, edgecolor='white', linewidth=0.8
    )
    axes[1].set_title(f'Depois do SMOTE +{SMOTE_PCT}%', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Classe (Cenário B)', fontsize=11)
    axes[1].set_ylabel('Número de amostras (treino)', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Classe {c}' for c in classes])
    axes[1].set_ylim(0, ylim_top)
    axes[1].yaxis.grid(True, linestyle='--', linewidth=0.6, color='grey', alpha=0.5)
    axes[1].yaxis.set_minor_locator(plt.MultipleLocator(100))
    axes[1].yaxis.grid(True, which='minor', linestyle=':', linewidth=0.4, color='grey', alpha=0.3)
    axes[1].set_axisbelow(True)
    for bar, val in zip(bars_after, [counts_after.get(c, 0) for c in classes]):
        pct = val / total_after * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                     f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Distribuição de Classes no Conjunto de Treino — Cenário B (5 Classes)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = 'graficos/smote_comparison_B.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Gráfico guardado em {path}")


if __name__ == '__main__':
    main()
