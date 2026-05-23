"""
model_svm.py — Support Vector Machine (SVC)
============================================
Implementa e optimiza o SVM para os 3 cenários de classificação do nível de stress:
  Cenário A — Binário (2 classes)
  Cenário B — Multiclasse (5 classes)
  Cenário C — Multiclasse (10 classes)

Decisões de implementação:
  - Pipeline(StandardScaler, SVC): previne data leakage na cross-validation
  - kernel='rbf': mais expressivo para dados não lineares; testado também 'linear'
  - class_weight='balanced': corrige automaticamente o desequilíbrio de classes
  - probability=True: necessário para calcular ROC-AUC
  - GridSearchCV com StratifiedKFold(k=5): busca exaustiva num espaço pequeno (18 combinações)

Referência: Cortes, C. & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
"""

import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

from código.modelos.utils import (
    compute_metrics, log_experiment, plot_confusion_matrix_fig,
    plot_cv_baseline, split_scenario, save_model, update_registry,
    CV_FOLDS, RANDOM_STATE
)
from código.data_pipeline import load_and_inspect_data, generate_targets, feature_engineering


# =============================================================
# PIPELINE BASE
# =============================================================
def _build_svm_pipeline():
    """
    Constrói a pipeline base para o modelo Support Vector Machine (SVC).
    Inclui o StandardScaler para normalização dos dados e o próprio SVC,
    com pesos equilibrados (class_weight='balanced') e cálculos de probabilidade ativos.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])


# =============================================================
# BASELINE (sem tuning)
# =============================================================
def train_svm_baseline(X_train, y_train, X_val, y_val, scenario):
    """
    Treina e avalia o modelo SVM com os hiperparâmetros por defeito (Baseline).
    Gera as predições no conjunto de validação, regista as métricas de desempenho
    e cria a matriz de confusão associada.
    """
    print(f"\n  [SVM Baseline] Cenário {scenario}")
    pipe = _build_svm_pipeline()
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)
    metrics = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Val F1 Macro: {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('SVM', scenario,
                   {'kernel': 'rbf', 'C': 1, 'gamma': 'scale', 'tuning': 'none'},
                   metrics, split='val_baseline')
    plot_confusion_matrix_fig(y_val, y_pred, 'SVM_baseline', scenario)
    return pipe, metrics


# =============================================================
# TUNING — GridSearchCV
# =============================================================
def tune_svm(X_train, y_train, X_val, y_val, scenario):
    """
    Optimiza os hiperparâmetros do SVM através de GridSearchCV com StratifiedKFold(k=5).
    Busca a melhor combinação de regularização (C), largura do kernel (gamma) e tipo de kernel,
    treinando o modelo final otimizado, avaliando no conjunto de validação e guardando a matriz de confusão.
    """
    print(f"\n  [SVM GridSearchCV] Cenário {scenario}")
    pipe = _build_svm_pipeline()
    param_grid = {
        'svm__C':      [0.1, 1, 10],
        'svm__gamma':  ['scale', 'auto'],
        'svm__kernel': ['rbf', 'linear'],
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, param_grid, cv=cv, scoring='f1_macro',
                          n_jobs=-1, refit=True)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    print(f"   Best params: {search.best_params_}  |  CV F1: {search.best_score_:.4f}")

    y_pred  = best.predict(X_val)
    y_proba = best.predict_proba(X_val)
    metrics = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Val F1 Macro (tuned): {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('SVM', scenario, search.best_params_, metrics, split='val_tuned')
    plot_confusion_matrix_fig(y_val, y_pred, 'SVM', scenario)
    return best, metrics


# =============================================================
# CV BASELINE
# =============================================================
def run_cv_baseline(X_raw, y, scenario, cv_rows):
    """
    Executa validação cruzada (Cross-Validation) no conjunto de dados completo com o modelo baseline.
    Regista as métricas médias (F1-macro) e desvios padrão, essenciais para ter uma métrica robusta
    da performance antes do tuning.
    """
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe  = _build_svm_pipeline()
    scores = cross_val_score(pipe, X_raw, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    mean_f1, std_f1 = scores.mean(), scores.std()
    print(f"   [CV] Cenário {scenario}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    cv_rows.append({'scenario': scenario, 'f1_mean': mean_f1, 'f1_std': std_f1})
    log_experiment('SVM', scenario,
                   {'cv_folds': CV_FOLDS, 'type': 'baseline_cv'},
                   {'f1_macro': mean_f1, 'f1_std': std_f1}, split='cv')
    return cv_rows


# =============================================================
# MAIN
# =============================================================
def main():
    """
    Função principal que orquestra o pipeline de treino do SVM:
    1. Carrega e pré-processa os dados.
    2. Executa o baseline com validação cruzada.
    3. Treina, optimiza, avalia e guarda os modelos finais para os cenários A, B e C.
    """
    print("=" * 60)
    print("   MODEL_SVM.PY — Support Vector Machine")
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
    plot_cv_baseline(cv_rows, 'SVM')

    # --- Per-scenario training ---
    registry = {}
    for sc, y in scenarios.items():
        print(f"\n{'='*60}\n   CENÁRIO {sc}\n{'='*60}")
        X_train, X_val, X_test, y_train, y_val, y_test = split_scenario(
            X_raw.values, y, sc
        )

        train_svm_baseline(X_train, y_train, X_val, y_val, sc)
        best_model, metrics = tune_svm(X_train, y_train, X_val, y_val, sc)

        save_model(best_model, f'svm_cenario_{sc}.pkl')
        registry[f'svm_{sc}'] = {'metrics_val': metrics}

    update_registry(registry)
    print("\n✅ model_svm.py concluído!")


if __name__ == '__main__':
    main()
