import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("Por favor, instale o imbalanced-learn localmente, caso falhe.")
    SMOTE, ADASYN, RandomUnderSampler, ImbPipeline = None, None, None, None

def load_and_inspect_data(filepath):
    """
    Carrega o dataset a partir de um ficheiro CSV, inspeciona o seu tamanho, 
    identifica e remove potenciais colunas irrelevantes (como IDs) e valores duplicados.
    Também exibe estatísticas descritivas e valores nulos.
    """
    print("--- 1.1 & 1.2 Importação, Inspecção e Limpeza ---")
    df_original = pd.read_csv(filepath)
    
    print("Shape do dataset (linhas, colunas):", df_original.shape)
    
    # 4. Identificar colunas irrelevantes (IDs, datas, etc.) antes das estatísticas
    irrelevant_cols = []
    for col in df_original.columns:
        if 'id' in col.lower():
            irrelevant_cols.append(col)
    
    if irrelevant_cols:
        print("\nColunas potencialmente irrelevantes:", irrelevant_cols)
        
    # 5. Estatísticas descritivas
    desc_cols = [c for c in df_original.columns if c not in irrelevant_cols]
    print("\nEstatísticas Descritivas (Amostra):")
    print(df_original[desc_cols].describe().iloc[:, :5])
    
    # Cópia segura e Limpeza
    df = copy.deepcopy(df_original)
    print("\nValores em Falta (Nulos)")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print(f"\nValores Duplicados detetados: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    
    return df

def generate_targets(df):
    """
    Gera as variáveis alvo (targets) para diferentes cenários de classificação:
    - Cenário A: Classificação binária (Com Stress / Sem Stress).
    - Cenário B: Classificação em 5 níveis de stress baseados em quantis.
    - Cenário C: Classificação em 10 níveis de stress (estudo com Equal-Width, Equal-Freq e K-Means).
    """
    print("\n--- 1.3 Transformação do Target (Cenários A, B e C) ---")
    
    # --- Cenário A ---
    limiar_A = 7.38
    df['target_A'] = np.where(df['stress_level'] >= limiar_A, 1, 0)
    print("\nDistribuição Cenário A (0 = Sem Stress, 1 = Com Stress):")
    print(df['target_A'].value_counts())
    
    # --- Cenário B ---
    df['target_B'] = pd.qcut(df['stress_level'], q=5, labels=False, duplicates='drop')
    print("\nDistribuição Cenário B (5 Classes - Numéricas 0 a 4):")
    print(df['target_B'].value_counts())
    
    # --- Cenário C (Estudo detalhado) ---
    nomes_10_classes = [f"Nivel_{i}" for i in range(1, 11)]
    
    # Método 1
    df['target_C_width'] = pd.cut(df['stress_level'], bins=10, labels=nomes_10_classes)
    
    # Método 2
    df['target_C_freq'] = pd.qcut(df['stress_level'], q=10, duplicates='drop')
    n_cats_freq = df['target_C_freq'].cat.categories.size
    nomes_freq = [f"Nivel_{i}" for i in range(1, n_cats_freq + 1)]
    nomes_dict_freq = {cat: nome for cat, nome in zip(df['target_C_freq'].cat.categories, nomes_freq)}
    df['target_C_freq'] = df['target_C_freq'].map(nomes_dict_freq)
    
    # Método 3
    kmeans = KMeans(n_clusters=10, random_state=42)
    grupos_kmeans = kmeans.fit_predict(df[['stress_level']])
    centros = kmeans.cluster_centers_.flatten()
    ordem = centros.argsort()
    mapa_ordem = {ordem[i]: f"Nivel_{i+1}" for i in range(10)}
    df['target_C_kmeans'] = [mapa_ordem[grupo] for grupo in grupos_kmeans]
    
    print("\n=== COMPARAÇÃO DO CENÁRIO C (10 Classes) ===\n")
    comparacao = pd.DataFrame({
        'Equal-Width': df['target_C_width'].value_counts(),
        'Equal-Freq': df['target_C_freq'].value_counts(),
        'K-Means': df['target_C_kmeans'].value_counts()
    }).fillna(0).astype(int)
    print(comparacao)
    
    # Atribuição da variável final numérica para pipeline e plotagem EDA
    df['target_C'] = pd.qcut(df['stress_level'], q=10, labels=False, duplicates='drop')
    
    return df

# 1.4 Análise Exploratória de Dados (EDA)
def run_eda(df):
    """
    Realiza a Análise Exploratória de Dados (EDA) e guarda os gráficos gerados na pasta 'graficos/'.
    Gera histogramas, gráficos de barras para as classes alvo, matrizes de correlação, 
    boxplots e pairplots para avaliar a distribuição e relação entre as variáveis.
    """
    print("\n--- 1.4 Análise Exploratória de Dados (EDA) ---")
    os.makedirs('graficos', exist_ok=True)
    
    # Isolar numéricas
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Ignorar columns desnecessárias dos targets secundários da EDA
    excluir = ['user_id', 'target_A', 'target_B', 'target_C', 'stress_level']
    excluir += [c for c in df.columns if 'target_C_' in c] 
    num_cols = [c for c in num_cols if c not in excluir]
    
    # 1. Distribuições univariadas
    plt.figure(figsize=(15, 12))
    for i, col in enumerate(num_cols):
        idx = min(i+1, 12)
        plt.subplot(4, 3, idx)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuição de {col}')
    plt.tight_layout()
    plt.savefig('graficos/eda_distribuicoes.png')
    plt.close()
    
    # 2. Target analysis e Class Imbalance
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(df['stress_level'], kde=True)
    plt.title('Stress Contínuo Original')
    
    plt.subplot(1, 3, 2)
    sns.countplot(x=df['target_B'], palette='Set2')
    plt.title('Target B (5 Classes)')
    
    plt.subplot(1, 3, 3)
    sns.countplot(x=df['target_C'], palette='Set3')
    plt.title('Target C (10 Classes)')
    plt.tight_layout()
    plt.savefig('graficos/eda_targets.png')
    plt.close()
    
    # 3. Matrizes de correlação (Pearson + Spearman lado a lado)
    corr_cols = num_cols + ['stress_level']
    pearson_corr = df[corr_cols].corr(method='pearson')
    spearman_corr = df[corr_cols].corr(method='spearman')

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0],
                annot_kws={"size": 7})
    axes[0].set_title('Correlação de Pearson\n(relações lineares)', fontsize=13)
    axes[0].tick_params(labelsize=7)
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1],
                annot_kws={"size": 7})
    axes[1].set_title('Correlação de Spearman\n(relações monotónicas)', fontsize=13)
    axes[1].tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig('graficos/eda_correlacao_combined.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Manter também versão Pearson isolada para compatibilidade
    corr = pearson_corr
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap - Correlação de Pearson com o Target')
    plt.savefig('graficos/eda_correlacao.png')
    plt.close()
    
    # 4. Box plots por classe (vs Cenário B)
    if len(num_cols) >= 2:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='target_B', y=num_cols[0], data=df)
        plt.title(f'{num_cols[0]} por Classe (Cenário B)')
        plt.savefig('graficos/eda_boxplot.png')
        plt.close()
    
        # 5. Pair plots das 3 variáveis mais correlacionadas com stress_level
        top_features = corr['stress_level'].abs().sort_values(ascending=False).index[1:4].tolist()
        sns.pairplot(df, vars=top_features, hue='target_B', palette='viridis')
        plt.savefig('graficos/eda_pairplot.png')
        plt.close()
    
    print("-> Gráficos da EDA guardados na pasta 'graficos/'.")

# 1.5 Feature Engineering
def feature_engineering(df):
    """
    Cria novas variáveis (features) com base nas existentes para potenciar os modelos preditivos.
    Calcula rácios (ex: uso de ecrã / sono), interações e variáveis binárias (ex: sedentário).
    Aplica codificação One-Hot nas variáveis categóricas e remove características altamente correlacionadas.
    """
    print("\n--- 1.5 Feature Engineering ---")
    df_fe = df.copy()
    epsilon = 1e-5
    
    # Rácios propostos
    df_fe['ratio_usage_sleep'] = df_fe['daily_screen_time_hours'] / (df_fe['sleep_duration_hours'] + epsilon)
    df_fe['ratio_activity_usage'] = (df_fe['physical_activity_minutes'] / 60.0) / (df_fe['daily_screen_time_hours'] + epsilon)
    
    # Interação entre features que podem potenciar o stress
    df_fe['interaction_usage_fatigue'] = df_fe['daily_screen_time_hours'] * df_fe['mental_fatigue_score']
    
    # Variáveis Binárias
    df_fe['dorme_menos_6h'] = np.where(df_fe['sleep_duration_hours'] < 6, 1, 0)
    df_fe['sedentario'] = np.where(df_fe['physical_activity_minutes'] < 30, 1, 0)
    
    # Identificar colunas para ignorar no modelo
    drop_cols = ['user_id', 'stress_level', 'target_A', 'target_B', 'target_C']
    drop_cols += [c for c in df_fe.columns if 'target_C_' in c]
    
    X_tmp = df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns])
    
    # One-Hot Encoding rápido das nominais
    categorical_cols = ['gender', 'occupation']
    X_enc = pd.get_dummies(X_tmp, columns=[c for c in categorical_cols if c in X_tmp.columns], drop_first=True)
    
    # Avaliar importâncias preliminares em relação ao Cenário A
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_enc, df_fe['target_A'])
    
    importances = pd.Series(rf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
    print("Importância Preliminar das Features (Top 5):")
    print(importances.head())
    
    # Filtro de Correlação (>0.95)
    corr_matrix = X_enc.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        print(f"Features removidas devido a correlação cruzada alta: {to_drop}")
        X_enc = X_enc.drop(columns=to_drop)
    
    return X_enc, df_fe

# 1.6 Pré-processamento Final e Divisão dos Dados
def preprocess_and_split(X, y):
    """
    Divide os dados em conjuntos de Treino (70%), Validação (15%) e Teste (15%) de forma estratificada.
    Aplica a normalização (StandardScaler) evitando fugas de dados (data leakage), ajustando apenas no treino.
    """
    print("\n--- 1.6 Pré-processamento e Divisão dos Dados ---")
    
    # Divisão: 70% treino, 15% validação, 15% teste
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), stratify=y_temp, random_state=42)
    
    print(f"Split - Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
    
    # 🔴 Importante: Data Leakage prevention
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Normalização com StandardScaler aplicada APENAS com fit_transform no Treino.")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

# 1.7 Estratégias para Desequilíbrio de Classes
def apply_imbalance_strategies(X_train, y_train):
    """
    Aplica estratégias de reamostragem (como o SMOTE) para lidar com o desequilíbrio das classes
    nos dados de treino, gerando amostras sintéticas para as classes minoritárias.
    """
    print("\n--- 1.7 Estratégias para Desequilíbrio de Classes (Treino) ---")
    if SMOTE is None:
        print("Módulos imbalanced-learn não configurados. Ignorando.")
        return
        
    print("Contagem Original de Classes:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # 1. SMOTE
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print("\n[SMOTE] Contagem de Classes:")
    print(pd.Series(y_sm).value_counts().sort_index())

def main():
    """
    Função principal que orquestra todo o pipeline de dados:
    Desde o carregamento do ficheiro até ao tratamento de desequilíbrios,
    passando por engenharia de features, EDA e divisão dos dados, deixando tudo preparado para a modelação.
    """
    print("==================================================")
    print("       PIPELINE FINAL COMPLETA (1.1 a 1.7)      ")
    print("==================================================")
    
    filepath = 'sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        print(f"Erro: O ficheiro '{filepath}' não foi encontrado.")
        return
        
    # 1.1 e 1.2 Importação e Limpeza (Código da Amiga)
    df = load_and_inspect_data(filepath)
    
    # 1.3 Target Transformation (Código da Amiga)
    df = generate_targets(df)
    
    # 1.4 EDA (Meu código)
    run_eda(df)
    
    # 1.5 Feature Engineering (Meu código)
    X_features, df_fe = feature_engineering(df)
    
    # Escolha de variável para 1.6 e 1.7
    # Usaremos Cenário B (5 classes)
    y_cenario_B = df_fe['target_B'].astype(int)
    
    # 1.6 Pré-processamento e Divisão
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = preprocess_and_split(X_features, y_cenario_B)
    
    # 1.7 Tratamento de Desequilíbrio
    apply_imbalance_strategies(X_train_s, y_train)
    
    print("\n✅ Pipeline base concluída com sucesso!")

if __name__ == '__main__':
    main()