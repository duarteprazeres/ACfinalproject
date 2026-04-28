# ACfinalproject

## Plano de Ação - Previsão de Stress com ML
Roteiro detalhado, revisto e melhorado para o desenvolvimento do projeto de Machine Learning — do dataset ao relatório final.

### Visão Geral do Projeto
**🎯 Objetivo**  
Treinar modelos de ML capazes de prever o nível de stress de indivíduos com base em dados de uso de smartphone, padrões de sono e atividade física.

**⚡ O "Twist"**  
A variável alvo é contínua. É necessário discretizá-la em três cenários de classificação distintos antes de treinar qualquer modelo.

**🗂️ Os Três Cenários de Classificação**
* **Cenário A:** Classificação Binária (2 classes: sem_stress / com_stress)
* **Cenário B:** 5 Classes (Sem stress → Baixo → Moderado → Alto → Extremo)
* **Cenário C:** 10 Classes (Nível 0 a Nível 9 (escala granular))

Todos os gráficos gerados pelo projeto devem aparecer na pasta graficos.
---

### Plano de Ação

#### FASE 0: Setup Inicial & Preparação do Ambiente
- [x] **0.1 Criar Repositório Git**: Inicializa um repositório Git.
- [x] **0.2 Configurar Ambiente Python**: Cria e ativa um ambiente virtual. Instala dependências (pandas, scikit-learn, etc).
- [x] **0.3 Download e Verificação do Dataset**: Descarrega do Kaggle. Inspecções iniciais.

#### FASE 1: Preparação e Exploração de Dados (EDA)
- [] **1.1 Importação e Inspecção Inicial dos Dados**: Carrega o CSV, verifica estrutura, identifica target.
- [ ] **1.2 Limpeza e Tratamento de Dados**: Trata nulos, outliers, inconsistências (IQR, fillna).
- [] **1.3 Transformação do Target — Criação dos 3 Cenários**: Cenários A, B e C aplicados (via função `create_target_scenarios()` no script `duarte_code_editor.py`, usando `np.where` e `pd.cut`).
- [x] **1.4 Análise Exploratória de Dados (EDA)**: Gráficos gerados e exportados para ficheiros `.png` na pasta local (distribuições, alvos, heatmap de correlação, boxplot, pairplot) através da função `run_eda()`.
- [x] **1.5 Feature Engineering**: Novas variáveis criadas (`ratio_usage_sleep`, `ratio_activity_usage`, interações, binárias clínicas como `dorme_menos_6h`). Seleção testada via feature importance de RandomForest preliminar e filtragem feita para correlações elevadas (>0.95), implementado na função `feature_engineering()`.
- [x] **1.6 Pré-processamento Final e Divisão dos Dados**: Dados separados em 70/15/15 com `stratify`. O `StandardScaler` foi ajustado apenas no conjunto de Treino para prevenir data leakage e depois aplicado de forma isolada sobre a Validação e Teste (via `preprocess_and_split()`).
- [x] **1.7 Estratégias para Desequilíbrio de Classes**: Métodos SMOTE, ADASYN e RandomUnderSampler implementados para observar e corrigir o class imbalance nos dados de treino (via `apply_imbalance_strategies()`).

#### FASE 2: Desenvolvimento dos Modelos e Experiências 
- [x] **2.1 Seleção e Justificação do 3.º Modelo**: **XGBoost** — ensemble tree-based, complementa SVM e NN, invariante à escala, interpretável via SHAP (Chen & Guestrin, KDD 2016).
- [x] **2.2 Definição das Métricas de Avaliação**: F1-Score (macro) como principal; Precision, Recall, ROC-AUC, MCC (A); Weighted-F1, Cohen's Kappa (B/C); Top-2 Accuracy (C).
- [x] **2.3 Implementação do SVM**: `Pipeline(StandardScaler + SVC)`, kernel RBF, `class_weight='balanced'`, `GridSearchCV` sobre C × gamma × kernel.
- [x] **2.4 Implementação das Redes Neuronais (NN)**: `MLPClassifier` (128→64, ReLU, Adam), L2 reg, EarlyStopping (patience=10), `RandomizedSearchCV`.
- [x] **2.5 Implementação do 3.º Modelo (XGBoost)**: `XGBClassifier`, `sample_weight='balanced'`, `eval_set` + `early_stopping_rounds`, `RandomizedSearchCV`.
- [x] **2.6 Cross-Validation e Baseline**: `StratifiedKFold(k=5)`, mean ± std F1 para todos os modelos × cenários, gráfico de barras exportado.
- [x] **2.7 Hyperparameter Tuning**: `GridSearchCV` (SVM), `RandomizedSearchCV` (NN, XGBoost); resultados registados em CSV.
- [x] **2.8 Logging de Experiências**: `experiment_results.csv` com timestamp, modelo, cenário, parâmetros e métricas por run.
- [x] **2.9 Persistência dos Modelos Finais**: `joblib.dump` → `models/` (9 pkl); `model_registry.json` com melhores params e métricas.

#### FASE 3: Análise de Erros e Avaliação Final 
- [ ] **3.1 Análise de Erros (Error Analysis)**
- [ ] **3.2 Interpretabilidade dos Modelos**: Extração de correlações e SHAP Values.
- [ ] **3.3 Testes de Significância Estatística**: Aplicar teste de McNemar / Wilcoxon para comparações.
- [ ] **3.4 Avaliação Final no Conjunto de Teste (Hold-out)**
- [ ] **3.5 Tabela Comparativa Final**

#### FASE 4: Documentação e Relatório Final
- [x] **4.1 Estrutura Obrigatória do PDF**
- [x] **4.2 Dicas para o Conteúdo Analítico do Relatório**
- [x] **4.3 AI Disclaimer Obrigatório** (Se aplicável)
- [x] **4.4 Checklist Final Antes de Submeter**

---

##### Referência Rápida — Modelos e Parâmetros
| Modelo | Obrigatório? | Normalização | Parâmetros Principais | Cenários |
|---|---|---|---|---|
| **SVM (SVC)** | SIM | Obrigatória | C, gamma, kernel | A, B, C |
| **Rede Neuronal** | SIM | Recomendada | layers, lr, dropout, batch_size | A, B, C |
| **XGBoost / LGBM** | RECOMENDADO | Não necessária | n_estimators, max_depth, learning_rate | A, B, C |
| **Random Forest** | ALTERNATIVA | Não necessária | n_estimators, max_depth, min_samples_leaf | A, B, C |
| **Logistic Regression**| ALTERNATIVA | Obrigatória | C, solver, multi_class | A (baseline) |

---

### Resumo das Preparações (Fase 1) - Base para o Relatório

Esta secção detalha as decisões técnicas e transformações processadas na base de dados durante as fases 1.1 a 1.7, preparando a informação para a modelação algorítmica.

#### 1.1 & 1.2: Inspecção e Limpeza de Dados
O conjunto de dados inicial contém 15.000 instâncias e 13 variáveis. Analisou-se primeiramente a exclusão de colunas não preditivas de identificação geométrica (como o `user_id`). Verificou-se não existirem valores nulos (missing values) ou registos estritamente duplicados, validando a integridade do dataset. Foram compiladas as estatísticas descritivas das variáveis (média, desvio padrão e percentis), consolidando uma base rigorosa para enquadrar a demografia e padrão base da nossa população alvo.

#### 1.3: Transformação do Target (Cenários A, B e C)
Para dar resposta aos problemas propostos, a variável preditiva subjacente (`stress_level`) precisou de ser separada:
- **Cenário A (Binário)**: Otimizou-se a fronteira baseada no limiar de divisão `>= 7.38`, construindo um binário razoavelmente mais equilibrado do que se dividíssemos na mediana natural. 
- **Cenário B (5 Classes)**: Aplicou-se um enquadramento por percentis (Equal-Frequency binning com *pd.qcut*) de modo a dividir uniformemente os graus de intensidade de stress pelas 5 esferas (Níveis 0 a 4).
- **Cenário C (10 Classes)**: Empregaram-se e avaliaram-se três métodos empíricos autónomos de "clustering" das 10 quantificações — nomeadamente distâncias k-Means, distâncias métricas lineares (Equal-Width) e dispersão de decil (Equal-frequency).

#### 1.4: Análise Exploratória de Dados (EDA)
Foram exportados gráficos sistemáticos e documentais para o repositório (`graficos/`):
- Mapas KDE/Histograma na extração do perfil estatístico das covariáveis numéricas e os respetivos comportamentos de target imbalance.
- Identificação de multicorrelações usando Mapas de Cor de Pearson (Heatmaps), relevando dependências fortes contra o stress. 
- Investigação em subgrupos com Box Plots condicionados ao cenário de targets multiclasse mais propensos e o isolamento dos cruzamentos críticos de maior importância (Scatters via Pairplots).

#### 1.5: Feature Engineering
Extraiu-se conhecimento oculto da estrutura gerando variáveis combinadas superiores aos preditores singulares da base de dados originais:
- **Rácios de Proporção**: Rácios entre `ecrã vs tempo a dormir` bem como percentagem de esforço motor sobre esse mesmo ecrã (`atividade / tempo_ecrã`).
- **Interações e Multiplos**: Cruzaram-se penalizações somadas de horas de ecrã mediante o peso do `mental_fatigue_score`.
- **Dicotomias de Alarme (Flags)**: Identificação puramente binária de grupos clínicos vulneráveis, nomeadamente sedentarismo (desporto inferir a 30 minutos) e privações de sono (sleep < 6h). Finalizou-se calculando o peso intrínseco (importance_feature) num ensaio com Random Forest base, servindo para retirar atributos excessivamente correlacionados (Pearson > 0.95).

#### 1.6: Pré-processamento e Isolamento (Data Leakage)
Fracionámos as instâncias com amostragem estratificada (`stratify`) garantindo que as percentagens relativas das classes raras foram copiadas para a matriz de 70% Treino e as vizinhanças dos 15% Validação / 15% Teste.  Implementou-se ainda o `StandardScaler()`, rigidamente forçado com o `fit()` em modo fechado unicamente no cofre de Treino. Qualquer valor da base de validação e de holdout será transformado na projeção desvio unicamente baseada no treino (minimizando integralmente o Data Leakage). 

#### 1.7: Atenuação de Desequilíbrio
Diagnosticaram-se assimetrias sensíveis com o fracionamento policategórico dos targets C e B. Inseriu-se código para teste unitário de mitigadores baseados na biblioteca `imbalanced-learn`:
- **Oversampling Sintético**: Foram testadas a extração de vizinhos SMOTE e de reponderação adversária ADASYN como fontes artificiais das instâncias subrepresentadas. 
- **Destruição Aleatória (Undersampling)**: Remoção estocástica para o mínimo múltiplo comum sob o RandomUnderSampler.
Os outputs gerados revelam métricas e distribuições preparadas para alimentar a estrutura principal dos testes dos próximos modelos SVM e Redes Neuronais.

#### Nota Técnica: Arquitetura de Código (Modularidade e `def main()`)
A estrutura final do código script `data_pipeline.py` abandonou a execução linear aberta do Python em prol de um design rigorosamente modular unificado pelo controlador `if __name__ == '__main__':`. Este paradigma foi essencial devido aos seguintes critérios do Mercado ML:
- **Prevenção Máxima de Data Leakage vs Variáveis Globais**: O encapsulamento lógico em blocos fechados (como ex: `def feature_engineering(df):`) garantiu o esvaziamento iminente das variáveis intermédias de memória (Scope local). Evitaram-se as armadilhas perigosas da persistência de variáveis globais que, num ambiente solto, poderiam contaminar acidentalmente o cruzamento final entre o Dataset de Validação e o Treino.
- **Importabilidade Robusta Segura**: A declaração no main bloqueou as auto-execuções. Qualquer módulo colega posterior, ou blocos em Jupyter Notebooks das próximas secções de Rede Neuronal poderão executar explicitamente comandos como `from data_pipeline import generate_targets` usufruindo da capacidade da função original sem provocar imediatamente a repetição temporal maciça dos gráficos de EDA ou simulação pesada do Oversampling.
- **Micro-gestão Otimizada de Testes**: A hierarquia dividida entre funções puras e a invocação agrupada tornou extremamente simples desligar ou ignorar etapas pesadas já computadas e validadas (comentando uma única linha de chamada na base sem perder um fio condutor para todos os próximos modelos do passo 2).

---

## Arquitectura e Organização do Projecto

Esta secção explica como o projecto está organizado internamente — para que serve cada ficheiro, que funções foram criadas, como se relacionam entre si e como a informação migra de um script para outro.

---

### Mapa de Ficheiros

```
ACfinalproject/
│
├── data_pipeline.py     ← FASE 1: carregamento, EDA, feature engineering, pré-proc.
├── utils.py             ← Caixa de ferramentas partilhada por todos os modelos
├── model_svm.py         ← FASE 2: SVM — Cenários A, B e C
├── model_nn.py          ← FASE 2: Rede Neuronal MLP — Cenários A, B e C
├── model_xgboost.py     ← FASE 2: XGBoost — Cenários A, B e C
├── evaluation_pipeline.py ← FASE 3: Análise de erros, testes estatísticos, testes finais
├── phase2.py            ← Script unificado da Fase 2 (versão anterior monolítica)
├── relatorio.tex / .pdf ← Relatório completo em LaTeX (14 páginas)
├── experiment_results.csv ← Registo automático de todas as experiências
│
├── models/
│   ├── svm_cenario_A/B/C.pkl
│   ├── nn_cenario_A/B/C.pkl
│   ├── xgb_cenario_A/B/C.pkl
│   ├── le_xgb_cenario_A/B/C.pkl   ← LabelEncoder do XGBoost
│   └── model_registry.json        ← Melhores params e métricas de cada modelo
│
├── graficos/            ← Todos os gráficos (EDA + confusion matrices + loss curves)
└── sleep_mobile_stress_dataset_15000.csv
```

---

### Como a Informação Flui Entre Scripts

```
CSV bruto
   │
   ▼
data_pipeline.py::feature_engineering()   ← devolve X (features) e df_fe (com targets)
   │
   │  from data_pipeline import load_and_inspect_data, generate_targets, feature_engineering
   ▼
model_svm.py / model_nn.py / model_xgboost.py
   │
   │  from utils import compute_metrics, log_experiment, split_scenario, save_model, ...
   ├── split_scenario()            → divide X e y em treino/val/teste (70/15/15)
   ├── compute_metrics()           → calcula F1, ROC-AUC, Kappa, MCC
   ├── log_experiment()            → escreve linha em experiment_results.csv
   ├── plot_confusion_matrix_fig() → guarda confusion matrix em graficos/
   └── save_model()                → guarda modelo em models/*.pkl
```

**Princípio fundamental:** nenhum código é duplicado. Se uma função já existe em `data_pipeline.py` ou `utils.py`, os outros scripts importam-na directamente.

---

### `data_pipeline.py` — Fase 1

| Função | O que faz |
|---|---|
| `load_and_inspect_data(filepath)` | Carrega o CSV, remove duplicados, mostra estatísticas |
| `generate_targets(df)` | Cria `target_A` (binário), `target_B` (5 cls), `target_C` (10 cls) |
| `run_eda(df)` | Gera histogramas, heatmaps, boxplots → `graficos/` |
| `feature_engineering(df)` | Cria novas variáveis, codifica categorias, remove correlações > 0.95. Devolve `X` e `df_fe` |
| `preprocess_and_split(X, y)` | Divide 70/15/15 com stratify; StandardScaler apenas no treino |
| `apply_imbalance_strategies(X, y)` | Demonstra SMOTE, ADASYN e RandomUnderSampler |

---

### `utils.py` — Utilitários Partilhados

| Função / Constante | O que faz |
|---|---|
| `RANDOM_STATE = 42` | Semente aleatória — resultados reproduzíveis |
| `CV_FOLDS = 5` | Número de folds na validação cruzada |
| `compute_metrics(y_true, y_pred, y_proba, scenario)` | Calcula todas as métricas; devolve dicionário |
| `log_experiment(model, scenario, params, metrics)` | Escreve linha no `experiment_results.csv` |
| `plot_confusion_matrix_fig(...)` | Gera e guarda confusion matrix normalizada |
| `plot_cv_baseline(cv_rows, model_name)` | Gráfico de barras F1 mean ± std |
| `split_scenario(X, y, scenario)` | Split estratificado 70/15/15 |
| `save_model(obj, filename)` | Guarda `.pkl` em `models/` |
| `update_registry(dict)` | Actualiza `model_registry.json` |

---

### `model_svm.py`

| Função | O que faz |
|---|---|
| `_build_svm_pipeline()` | `Pipeline(StandardScaler + SVC)` com `class_weight='balanced'` |
| `train_svm_baseline(...)` | Treina com parâmetros default; avalia; loga no CSV |
| `tune_svm(...)` | **GridSearchCV**: `C` × `gamma` × `kernel` = 12 combinações × 5 folds |
| `run_cv_baseline(...)` | `StratifiedKFold(5)` → F1 mean ± std |
| `main()` | Orquestra tudo para cenários A, B, C |

Usa **GridSearchCV** (exaustivo) porque o espaço de parâmetros é pequeno (12 combinações).

---

### `model_nn.py`

| Função | O que faz |
|---|---|
| `_build_nn_pipeline()` | `Pipeline(StandardScaler + MLPClassifier)` — 128→64 neurónios, ReLU, Adam |
| `plot_nn_loss(mlp, scenario, tag)` | Gráfico de loss por época guardado em `graficos/` |
| `train_nn_baseline(...)` | Treina com arquitectura base; gera loss curve |
| `tune_nn(...)` | **RandomizedSearchCV** (15 tentativas): arquitectura, alpha, lr, batch |
| `run_cv_baseline(...)` | `StratifiedKFold(5)` |
| `main()` | A, B, C → guarda `nn_cenario_*.pkl` |

Usa **RandomizedSearchCV** (aleatório) porque o espaço é grande (72+ combinações).

---

### `model_xgboost.py`

| Função | O que faz |
|---|---|
| `_build_xgb(n_classes)` | `XGBClassifier` — `binary:logistic` para A, `multi:softprob` para B e C |
| `train_xgb_baseline(...)` | Treina com `eval_set` + `early_stopping_rounds`; avalia |
| `tune_xgb(...)` | **RandomizedSearchCV** (20 tentativas): `n_estimators`, `max_depth`, `learning_rate`, etc. |
| `run_cv_baseline(...)` | Inclui `LabelEncoder` — XGBoost requer labels 0..N-1 |
| `main()` | A, B, C → guarda `xgb_cenario_*.pkl` + `le_xgb_cenario_*.pkl` |

---

### O que são os ficheiros `.pkl`?

`.pkl` (pickle) guarda qualquer objecto Python em disco. É como tirar uma fotografia ao modelo treinado:

```python
# Carregar e usar (Fase 3 — avaliação no conjunto de teste)
import joblib

# SVM e NN: o Pipeline já inclui o StandardScaler
model = joblib.load('models/svm_cenario_A.pkl')
predicoes = model.predict(X_teste)

# XGBoost: necessário também o LabelEncoder
xgb_model = joblib.load('models/xgb_cenario_B.pkl')
le         = joblib.load('models/le_xgb_cenario_B.pkl')
pred_enc   = xgb_model.predict(X_teste)
predicoes  = le.inverse_transform(pred_enc)
```

Os `.pkl` são usados na **Fase 3** para avaliar os modelos no conjunto de teste (hold-out), sem necessidade de re-treinar.

---

### Como Correr o Projecto

```bash
source venv/bin/activate   # activar ambiente virtual

python data_pipeline.py    # Fase 1 — dados e EDA
python model_svm.py        # Fase 2 — SVM
python model_nn.py         # Fase 2 — Rede Neuronal
python model_xgboost.py    # Fase 2 — XGBoost
python evaluation_pipeline.py # Fase 3 — Avaliação e Error Analysis

pdflatex relatorio.tex     # Compilar relatório PDF
```

Cada script de modelo é **completamente independente** — podes correr apenas um sem precisar de correr os outros.
