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
- [ ] **2.1 Seleção e Justificação do 3.º Modelo**: XGBoost, LightGBM, Random Forest, etc.
- [ ] **2.2 Definição das Métricas de Avaliação**: F1-Score (macro) como principal.
- [ ] **2.3 Implementação do SVM**
- [ ] **2.4 Implementação das Redes Neuronais (NN)**
- [ ] **2.5 Implementação do 3.º Modelo (ex: XGBoost/LightGBM)**
- [ ] **2.6 Cross-Validation e Baseline**: Usar StratifiedKFold.
- [ ] **2.7 Hyperparameter Tuning**: Otimizar com GridSearchCV / RandomizedSearchCV ou Optuna.
- [ ] **2.8 Logging de Experiências**: Registo sistemático num CSV ou MLflow.
- [ ] **2.9 Persistência dos Modelos Finais**: Guardar (`joblib.dump`) os modelos finais e o scaler.

#### FASE 3: Análise de Erros e Avaliação Final 
- [ ] **3.1 Análise de Erros (Error Analysis)**
- [ ] **3.2 Interpretabilidade dos Modelos**: Extração de correlações e SHAP Values.
- [ ] **3.3 Testes de Significância Estatística**: Aplicar teste de McNemar / Wilcoxon para comparações.
- [ ] **3.4 Avaliação Final no Conjunto de Teste (Hold-out)**
- [ ] **3.5 Tabela Comparativa Final**

#### FASE 4: Documentação e Relatório Final
- [ ] **4.1 Estrutura Obrigatória do PDF**
- [ ] **4.2 Dicas para o Conteúdo Analítico do Relatório**
- [ ] **4.3 AI Disclaimer Obrigatório** (Se aplicável)
- [ ] **4.4 Checklist Final Antes de Submeter**

---

##### Referência Rápida — Modelos e Parâmetros
| Modelo | Obrigatório? | Normalização | Parâmetros Principais | Cenários |
|---|---|---|---|---|
| **SVM (SVC)** | SIM | Obrigatória | C, gamma, kernel | A, B, C |
| **Rede Neuronal** | SIM | Recomendada | layers, lr, dropout, batch_size | A, B, C |
| **XGBoost / LGBM** | RECOMENDADO | Não necessária | n_estimators, max_depth, learning_rate | A, B, C |
| **Random Forest** | ALTERNATIVA | Não necessária | n_estimators, max_depth, min_samples_leaf | A, B, C |
| **Logistic Regression**| ALTERNATIVA | Obrigatória | C, solver, multi_class | A (baseline) |
