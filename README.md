# Classificação do Nível de Stress

**Projeto Final — Aprendizagem Computacional (LIACD 2025/2026)**  
Duarte Prazeres · Marília Brito

---

## Descrição

Classificação do nível de stress de indivíduos com base em dados comportamentais de smartphone (tempo de ecrã, sono, atividade física, fadiga). O dataset contém 15 000 registos sintéticos.

São abordados três cenários de classificação:

| Cenário | Tipo | Classes |
|---------|------|---------|
| **A** | Binário | 2 (sem stress / com stress) |
| **B** | Multiclasse | 5 (equal-width binning) |
| **C** | Multiclasse | 10 (equal-width binning) |

---

## Modelos

| Modelo | Biblioteca | Tuning |
|--------|-----------|--------|
| SVM | scikit-learn | GridSearchCV (12 combinações) |
| Rede Neuronal (MLP) | scikit-learn | RandomizedSearchCV (15 iterações) |
| XGBoost | xgboost | RandomizedSearchCV (20 iterações) |
| Ensemble de RN | PyTorch | 5 redes com arquiteturas diversas |

---

## Estrutura do Repositório

```
ACfinalproject/
│
├── código/
│   ├── data_pipeline.py          ← Carregamento, EDA, feature engineering, SMOTE
│   ├── evaluation_pipeline.py    ← Avaliação final, testes estatísticos, SHAP
│   ├── plot_smote_comparison.py  ← Gráfico comparativo antes/depois SMOTE
│   └── modelos/
│       ├── utils.py              ← Métricas, logging, gráficos (partilhado)
│       ├── model_svm.py          ← SVM — Cenários A, B, C
│       ├── model_nn.py           ← MLP sklearn — Cenários A, B, C
│       ├── model_xgboost.py      ← XGBoost — Cenários A, B, C
│       └── model_nn_ensemble.py  ← Ensemble PyTorch — Cenários A, B, C
│
├── models/
│   ├── svm_cenario_{A,B,C}.pkl
│   ├── nn_cenario_{A,B,C}.pkl
│   ├── xgb_cenario_{A,B,C}.pkl
│   ├── le_xgb_cenario_{A,B,C}.pkl   ← LabelEncoder do XGBoost
│   ├── nn_ensemble_{A,B,C}_model{0-4}.pt  ← Redes PyTorch
│   ├── nn_ensemble_scaler_{A,B,C}.pkl
│   ├── model_registry.json           ← Métricas e params dos modelos finais
│   └── experiment_results.csv        ← Log de todas as experiências
│
├── graficos/                     ← Todos os gráficos gerados (EDA, CM, loss curves, etc.)
├── dataset/
│   └── sleep_mobile_stress_dataset_15000.csv
├── relatório/
│   ├── relatorio.tex
│   └── relatorio.pdf
│
├── requirements.txt
└── README.md
```

---

## Como Correr

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Correr o pipeline completo (todas as fases de uma vez)
python3 run.py

# Ou correr cada fase individualmente:
python3 -m código.data_pipeline                  # Fase 1 — Dados, EDA e feature engineering
python3 -m código.modelos.model_svm              # Fase 2 — SVM
python3 -m código.modelos.model_nn               # Fase 2 — MLP
python3 -m código.modelos.model_xgboost          # Fase 2 — XGBoost
python3 -m código.modelos.model_nn_ensemble      # Fase 2 — Ensemble PyTorch
python3 -m código.evaluation_pipeline            # Fase 3 — Avaliação final

# 3. Compilar relatório PDF
cd relatório && pdflatex relatorio.tex && pdflatex relatorio.tex
```

---

## Resultados Principais (Conjunto de Teste)

| Modelo | Cenário A (F1) | Cenário B (F1) | Cenário C (F1) |
|--------|---------------|---------------|---------------|
| NN/MLP | **0.9502** | 0.7504 | 0.4994 |
| SVM | 0.9476 | **0.7526** | **0.5178** |
| XGBoost | 0.9462 | 0.7519 | 0.4991 |
| Ensemble PyTorch | 0.9489 | **0.7572** | 0.4932 |

---

## Dependências Principais

- `scikit-learn`, `xgboost`, `torch`, `imbalanced-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `shap`, `joblib`

Ver `requirements.txt` para versões exatas.
