"""
run.py — Pipeline completo do projeto
======================================
Executa todas as fases em sequência:
  1. Dados, EDA e feature engineering
  2. Treino dos modelos (SVM, MLP, XGBoost, Ensemble PyTorch)
  3. Avaliação final e testes estatísticos

Uso: python3 run.py
"""

import time

from código.data_pipeline import main as run_data_pipeline
from código.modelos.model_svm import main as run_svm
from código.modelos.model_nn import main as run_nn
from código.modelos.model_xgboost import main as run_xgboost
from código.modelos.model_nn_ensemble import main as run_nn_ensemble
from código.evaluation_pipeline import main as run_evaluation


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    t0 = time.time()

    section("FASE 1 — Dados, EDA e Feature Engineering")
    run_data_pipeline()

    section("FASE 2 — Treino dos Modelos")
    run_svm()
    run_nn()
    run_xgboost()
    run_nn_ensemble()

    section("FASE 3 — Avaliação Final e Testes Estatísticos")
    run_evaluation()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline concluído em {elapsed / 60:.1f} min")
    print(f"{'=' * 60}\n")
