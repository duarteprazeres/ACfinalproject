"""
model_nn.py — Rede Neuronal (MLPClassifier)
============================================
Implementa e optimiza a Rede Neuronal para os 3 cenários de classificação:
  Cenário A — Binário (2 classes)
  Cenário B — Multiclasse (5 classes)
  Cenário C — Multiclasse (10 classes)

Decisões de implementação:
  - MLPClassifier (sklearn): integração nativa com Pipeline e RandomizedSearchCV
  - Arquitectura base: Input → 128(ReLU) → 64(ReLU) → Output(Softmax)
  - Solver Adam com lr=0.001: convergência rápida e adaptativa
  - alpha (L2): regularização equivalente ao Dropout em termos de efeito de generalização
  - EarlyStopping (patience=10): evita overfitting, pára quando val não melhora
  - RandomizedSearchCV (15 iter.): mais eficiente que GridSearch para espaços grandes

Referência: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.
            Rumelhart, D. et al. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from pathlib import Path

from código.modelos.utils import (
    compute_metrics, log_experiment, plot_confusion_matrix_fig,
    plot_cv_baseline, split_scenario, save_model, update_registry,
    CV_FOLDS, RANDOM_STATE, GRAFICOS_DIR
)
from código.data_pipeline import load_and_inspect_data, generate_targets, feature_engineering


# =============================================================
# PIPELINE BASE
# =============================================================
def _build_nn_pipeline():
    """
    Constrói a pipeline base para a Rede Neuronal (MLPClassifier).
    Inclui normalização (StandardScaler) e um classificador Multi-Layer Perceptron
    configurado com duas camadas ocultas, otimizador Adam, early stopping (para evitar overfitting)
    e um limite máximo de 200 épocas.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            alpha=1e-4,
            batch_size=32,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
            verbose=False
        ))
    ])


# =============================================================
# LOSS CURVE
# =============================================================
def plot_nn_loss(mlp_step, scenario_name, tag='tuned'):
    """
    Gera e guarda um gráfico com a evolução da função de perda (Loss Curve) da Rede Neuronal ao longo das épocas.
    Mostra a perda no conjunto de treino e, se aplicável, o score no conjunto de validação interno.
    """
    if not hasattr(mlp_step, 'loss_curve_') or not mlp_step.loss_curve_:
        return
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(mlp_step.loss_curve_, color='#4C72B0', label='Train Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss', color='#4C72B0')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')

    if hasattr(mlp_step, 'validation_scores_') and mlp_step.validation_scores_:
        ax2 = ax1.twinx()
        ax2.plot(mlp_step.validation_scores_, color='#DD8452',
                 linestyle='--', label='Val Score')
        ax2.set_ylabel('Val Score', color='#DD8452')
        ax2.tick_params(axis='y', labelcolor='#DD8452')

    ax1.set_title(f'MLP Loss Curve ({tag}) — Cenário {scenario_name}')
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    plt.tight_layout()
    path = GRAFICOS_DIR / f'loss_curve_nn_{scenario_name}_{tag}.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   -> Loss curve: {path}")


# =============================================================
# BASELINE
# =============================================================
def train_nn_baseline(X_train, y_train, X_val, y_val, scenario):
    """
    Treina e avalia a Rede Neuronal com a arquitetura e hiperparâmetros predefinidos (Baseline).
    Gera predições, calcula as métricas, guarda o log de resultados, cria a matriz de confusão
    e o gráfico da loss curve inicial.
    """
    print(f"\n  [NN Baseline] Cenário {scenario}")
    pipe = _build_nn_pipeline()
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)
    metrics = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Val F1 Macro: {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('NN_MLP', scenario,
                   {'hidden_layer_sizes': '(128,64)', 'alpha': 1e-4, 'tuning': 'none'},
                   metrics, split='val_baseline')
    plot_confusion_matrix_fig(y_val, y_pred, 'NN_MLP_baseline', scenario)
    plot_nn_loss(pipe.named_steps['mlp'], scenario, tag='baseline')
    return pipe, metrics


# =============================================================
# TUNING — RandomizedSearchCV
# =============================================================
def tune_nn(X_train, y_train, X_val, y_val, scenario):
    """
    Optimiza a Rede Neuronal usando RandomizedSearchCV devido ao amplo espaço de hiperparâmetros.
    Realiza 15 iterações de teste alterando a arquitetura da rede (hidden_layer_sizes),
    a taxa de regularização (alpha), a taxa de aprendizagem inicial e o tamanho do batch.
    Retorna o modelo otimizado e guarda as matrizes de confusão e curvas de erro.
    """
    print(f"\n  [NN RandomizedSearchCV] Cenário {scenario}")
    pipe = _build_nn_pipeline()
    param_dist = {
        'mlp__hidden_layer_sizes': [(128, 64), (256, 128, 64), (64, 32)],
        'mlp__alpha':              [1e-4, 1e-3, 1e-2],
        'mlp__learning_rate_init': [1e-3, 5e-4, 1e-4],
        'mlp__batch_size':         [32, 64],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=15, cv=cv,
                                scoring='f1_macro', n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"   Best params: {search.best_params_}  |  CV F1: {search.best_score_:.4f}")

    y_pred  = best.predict(X_val)
    y_proba = best.predict_proba(X_val)
    metrics = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Val F1 Macro (tuned): {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('NN_MLP', scenario, search.best_params_, metrics, split='val_tuned')
    plot_confusion_matrix_fig(y_val, y_pred, 'NN_MLP', scenario)
    plot_nn_loss(best.named_steps['mlp'], scenario, tag='tuned')
    return best, metrics


# =============================================================
# CV BASELINE
# =============================================================
def run_cv_baseline(X_raw, y, scenario, cv_rows):
    """
    Executa validação cruzada (Cross-Validation) de 5 dobras sobre o modelo baseline,
    mas com restrições mais fortes (max_iter=100) para acelerar o processo.
    Regista o F1-macro médio e o desvio padrão como referência base para o modelo.
    """
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = _build_nn_pipeline()
    pipe.set_params(mlp__max_iter=100, mlp__n_iter_no_change=5)
    scores    = cross_val_score(pipe, X_raw, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    mean_f1, std_f1 = scores.mean(), scores.std()
    print(f"   [CV] Cenário {scenario}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    cv_rows.append({'scenario': scenario, 'f1_mean': mean_f1, 'f1_std': std_f1})
    log_experiment('NN_MLP', scenario,
                   {'cv_folds': CV_FOLDS, 'type': 'baseline_cv'},
                   {'f1_macro': mean_f1, 'f1_std': std_f1}, split='cv')
    return cv_rows


# =============================================================
# MAIN
# =============================================================
def main():
    """
    Função principal que orquestra o pipeline de treino da Rede Neuronal:
    1. Carrega e prepara os dados.
    2. Executa a validação cruzada do baseline.
    3. Treina, otimiza e regista os modelos finais para todos os cenários.
    """
    print("=" * 60)
    print("   MODEL_NN.PY — Rede Neuronal (MLPClassifier)")
    print("=" * 60)

    filepath = 'dataset/sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        filepath = 'sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        print(f"Erro: ficheiro CSV não encontrado.")
        return

    df           = load_and_inspect_data(filepath)
    df           = generate_targets(df)
    X_raw, df_fe = feature_engineering(df)

    valid    = ~df_fe[['target_A', 'target_B_ew', 'target_C_ew']].isnull().any(axis=1)
    X_raw    = X_raw[valid].reset_index(drop=True)
    df_fe    = df_fe[valid].reset_index(drop=True)

    scenarios = {
        'A': df_fe['target_A'].astype(int),
        'B': df_fe['target_B_ew'].astype(int),
        'C': df_fe['target_C_ew'].astype(int),
    }

    # --- CV Baseline ---
    print("\n--- Cross-Validation Baseline ---")
    cv_rows = []
    for sc, y in scenarios.items():
        cv_rows = run_cv_baseline(X_raw.values, y, sc, cv_rows)
    plot_cv_baseline(cv_rows, 'NN_MLP')

    # --- Per-scenario training ---
    registry = {}
    for sc, y in scenarios.items():
        print(f"\n{'='*60}\n   CENÁRIO {sc}\n{'='*60}")
        X_train, X_val, X_test, y_train, y_val, y_test = split_scenario(
            X_raw.values, y, sc
        )

        train_nn_baseline(X_train, y_train, X_val, y_val, sc)
        best_model, metrics = tune_nn(X_train, y_train, X_val, y_val, sc)

        save_model(best_model, f'nn_cenario_{sc}.pkl')
        registry[f'nn_{sc}'] = {'metrics_val': metrics}

    update_registry(registry)
    print("\n✅ model_nn.py concluído!")


if __name__ == '__main__':
    main()
