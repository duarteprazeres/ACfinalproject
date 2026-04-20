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
A estrutura final do código script `main.py` abandonou a execução linear aberta do Python em prol de um design rigorosamente modular unificado pelo controlador `if __name__ == '__main__':`. Este paradigma foi essencial devido aos seguintes critérios do Mercado ML:
- **Prevenção Máxima de Data Leakage vs Variáveis Globais**: O encapsulamento lógico em blocos fechados (como ex: `def feature_engineering(df):`) garantiu o esvaziamento iminente das variáveis intermédias de memória (Scope local). Evitaram-se as armadilhas perigosas da persistência de variáveis globais que, num ambiente solto, poderiam contaminar acidentalmente o cruzamento final entre o Dataset de Validação e o Treino.
- **Importabilidade Robusta Segura**: A declaração no main bloqueou as auto-execuções. Qualquer módulo colega posterior, ou blocos em Jupyter Notebooks das próximas secções de Rede Neuronal poderão executar explicitamente comandos como `from main import generate_targets` usufruindo da capacidade da função original sem provocar imediatamente a repetição temporal maciça dos gráficos de EDA ou simulação pesada do Oversampling.
- **Micro-gestão Otimizada de Testes**: A hierarquia dividida entre funções puras e a invocação agrupada tornou extremamente simples desligar ou ignorar etapas pesadas já computadas e validadas (comentando uma única linha de chamada na base sem perder um fio condutor para todos os próximos modelos do passo 2).
