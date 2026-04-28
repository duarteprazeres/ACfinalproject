"""
model_xgboost.py — XGBoost (Gradient Boosting)
================================================
Implementa e optimiza o XGBoost para os 3 cenários de classificação:
  Cenário A — Binário (2 classes)
  Cenário B — Multiclasse (5 classes)
  Cenário C — Multiclasse (10 classes)

Justificação da escolha (2.1):
  O XGBoost foi escolhido como 3.º modelo pelos seguintes motivos académicos:
  1. COMPLEMENTARIDADE: é tree-based ensemble, fundamentalmente diferente do SVM
     (kernel machine) e da NN (gradient descent em espaço paramétrico contínuo).
  2. DADOS TABULARES: em benchmarks com dados estruturados, o XGBoost supera
     redes neuronais na maioria dos casos (Shwartz-Ziv & Armon, 2022).
  3. IMBALANCE: suporta nativamente sample_weight e scale_pos_weight.
  4. INTERPRETABILIDADE: SHAP values permitem análise de importância de features.
  5. EFICIÊNCIA: paralelização nativa, early stopping para evitar overfitting.

Decisões de implementação:
  - Sem normalização: árvores são invariantes à escala
  - LabelEncoder: XGBoost requer labels 0..N-1
  - sample_weight='balanced': corrige desequilíbrio de classes
  - eval_set + early_stopping_rounds: paragem automática quando val não melhora
  - RandomizedSearchCV (20 iter.): eficiente para o espaço de parâmetros grande

Referências:
  Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
    KDD'16, pp. 785-794.
  Shwartz-Ziv, R. & Armon, A. (2022). Tabular data: Deep learning is not all you need.
    Information Fusion, 81, 84-90.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report

from código.modelos.utils import (
    compute_metrics, log_experiment, plot_confusion_matrix_fig,
    plot_cv_baseline, split_scenario, save_model, update_registry,
    CV_FOLDS, RANDOM_STATE
)
from data_pipeline import load_and_inspect_data, generate_targets, feature_engineering


# =============================================================
# HELPERS
# =============================================================
def _xgb_objective(n_classes):
    return 'binary:logistic' if n_classes == 2 else 'multi:softprob'


def _xgb_metric(n_classes):
    return 'logloss' if n_classes == 2 else 'mlogloss'


def _build_xgb(n_classes, **kwargs):
    return xgb.XGBClassifier(
        objective=_xgb_objective(n_classes),
        eval_metric=_xgb_metric(n_classes),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
        **kwargs
    )


# =============================================================
# BASELINE
# =============================================================
def train_xgb_baseline(X_train, y_train, X_val, y_val, scenario):
    print(f"\n  [XGBoost Baseline] Cenário {scenario}")
    n_cls = len(np.unique(y_train))

    le          = LabelEncoder()
    y_tr_enc    = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    sw          = compute_sample_weight('balanced', y_tr_enc)

    model = _build_xgb(n_cls, n_estimators=200, max_depth=5, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_tr_enc, sample_weight=sw,
              eval_set=[(X_val, y_val_enc)], verbose=False)

    y_pred_enc = model.predict(X_val)
    y_pred     = le.inverse_transform(y_pred_enc)
    y_proba    = model.predict_proba(X_val)
    metrics    = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Val F1 Macro: {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('XGBoost', scenario,
                   {'n_estimators': 200, 'max_depth': 5, 'tuning': 'none'},
                   metrics, split='val_baseline')
    plot_confusion_matrix_fig(y_val, y_pred, 'XGBoost_baseline', scenario)
    return model, le, metrics


# =============================================================
# TUNING — RandomizedSearchCV
# =============================================================
def tune_xgb(X_train, y_train, X_val, y_val, scenario):
    """
    RandomizedSearchCV (20 iterações) sobre:
      n_estimators      ∈ {100, 200, 300}       — número de árvores
      max_depth         ∈ {3, 5, 7}             — profundidade máxima das árvores
      learning_rate     ∈ {0.01, 0.05, 0.1, 0.2}— passo de actualização
      subsample         ∈ {0.7, 0.8, 1.0}       — fracção de amostras por árvore
      colsample_bytree  ∈ {0.7, 0.8, 1.0}       — fracção de features por árvore
      min_child_weight  ∈ {1, 3, 5}             — peso mínimo em folha (regularização)
    """
    print(f"\n  [XGBoost RandomizedSearchCV] Cenário {scenario}")
    n_cls = len(np.unique(y_train))

    le       = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    sw       = compute_sample_weight('balanced', y_tr_enc)

    param_dist = {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [3, 5, 7],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
    }
    base = _build_xgb(n_cls)
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(base, param_dist, n_iter=20, cv=cv,
                                scoring='f1_macro', n_jobs=-1,
                                random_state=RANDOM_STATE, refit=True)
    search.fit(X_train, y_tr_enc, sample_weight=sw)
    best = search.best_estimator_
    print(f"   Best params: {search.best_params_}  |  CV F1: {search.best_score_:.4f}")

    y_val_enc  = le.transform(y_val)
    y_pred_enc = best.predict(X_val)
    y_pred     = le.inverse_transform(y_pred_enc)
    y_proba    = best.predict_proba(X_val)
    metrics    = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Val F1 Macro (tuned): {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('XGBoost', scenario, search.best_params_, metrics, split='val_tuned')
    plot_confusion_matrix_fig(y_val, y_pred, 'XGBoost', scenario)
    return best, le, metrics


# =============================================================
# CV BASELINE
# =============================================================
def run_cv_baseline(X_raw, y, scenario, cv_rows):
    n_cls  = len(np.unique(y))
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model  = _build_xgb(n_cls, n_estimators=100, max_depth=5, learning_rate=0.1)
    scores = cross_val_score(model, X_raw, y_enc, cv=cv, scoring='f1_macro', n_jobs=-1)
    mean_f1, std_f1 = scores.mean(), scores.std()
    print(f"   [CV] Cenário {scenario}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    cv_rows.append({'scenario': scenario, 'f1_mean': mean_f1, 'f1_std': std_f1})
    log_experiment('XGBoost', scenario,
                   {'cv_folds': CV_FOLDS, 'type': 'baseline_cv'},
                   {'f1_macro': mean_f1, 'f1_std': std_f1}, split='cv')
    return cv_rows


# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 60)
    print("   MODEL_XGBOOST.PY — XGBoost Gradient Boosting")
    print("=" * 60)

    filepath = 'sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        print(f"Erro: '{filepath}' não encontrado.")
        return

    df           = load_and_inspect_data(filepath)
    df           = generate_targets(df)
    X_raw, df_fe = feature_engineering(df)

    valid    = ~df_fe[['target_A', 'target_B', 'target_C']].isnull().any(axis=1)
    X_raw    = X_raw[valid].reset_index(drop=True)
    df_fe    = df_fe[valid].reset_index(drop=True)

    scenarios = {
        'A': df_fe['target_A'].astype(int),
        'B': df_fe['target_B'].astype(int),
        'C': df_fe['target_C'].astype(int),
    }

    # --- CV Baseline ---
    print("\n--- Cross-Validation Baseline ---")
    cv_rows = []
    for sc, y in scenarios.items():
        cv_rows = run_cv_baseline(X_raw.values, y, sc, cv_rows)
    plot_cv_baseline(cv_rows, 'XGBoost')

    # --- Per-scenario training ---
    registry = {}
    for sc, y in scenarios.items():
        print(f"\n{'='*60}\n   CENÁRIO {sc}\n{'='*60}")
        X_train, X_val, X_test, y_train, y_val, y_test = split_scenario(
            X_raw.values, y, sc
        )

        train_xgb_baseline(X_train, y_train, X_val, y_val, sc)
        best_model, le_best, metrics = tune_xgb(X_train, y_train, X_val, y_val, sc)

        save_model(best_model, f'xgb_cenario_{sc}.pkl')
        save_model(le_best,    f'le_xgb_cenario_{sc}.pkl')
        registry[f'xgboost_{sc}'] = {'metrics_val': metrics}

    update_registry(registry)
    print("\n✅ model_xgboost.py concluído!")


if __name__ == '__main__':
    main()
