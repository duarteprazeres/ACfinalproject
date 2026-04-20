import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def create_target_scenarios(df):
    """
    Cria as variáveis target para os Cenários A (2 classes), B (5 classes) e C (10 classes).
    Lida com o 'stress_level' contínuo (0 a 10).
    """
    df_targets = df.copy()
    
    # Cenário A: Binary (e.g., [0, 5[ -> 0, [5, 10] -> 1)
    df_targets['target_A'] = np.where(df_targets['stress_level'] < 5.0, 0, 1)
    
    # Cenário B: 5 classes
    df_targets['target_B'] = pd.cut(df_targets['stress_level'], bins=5, labels=False, include_lowest=True)
    
    # Cenário C: 10 classes
    df_targets['target_C'] = pd.cut(df_targets['stress_level'], bins=10, labels=False, include_lowest=True)
    
    return df_targets

# 1.4 Análise Exploratória de Dados (EDA)
def run_eda(df):
    print("--- 1.4 Análise Exploratória de Dados (EDA) ---")
    os.makedirs('graficos', exist_ok=True)
    df = create_target_scenarios(df)
    
    # Isolar numéricas
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['user_id', 'target_A', 'target_B', 'target_C', 'stress_level']]
    
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
    
    # 3. Matriz de correlação
    plt.figure(figsize=(10, 8))
    # Para o heatmap usamos as features numéricas + target original
    corr = df[num_cols + ['stress_level']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap - Correlação com o Target')
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
    print("\n--- 1.5 Feature Engineering ---")
    df_fe = df.copy()
    
    # Adicionar epsilon para evitar divisões por zero
    epsilon = 1e-5
    
    # Rácios propostos
    df_fe['ratio_usage_sleep'] = df_fe['daily_screen_time_hours'] / (df_fe['sleep_duration_hours'] + epsilon)
    # A atividade física está em minutos; dividimos por 60 hr para ficar tudo em horas ou comparável
    df_fe['ratio_activity_usage'] = (df_fe['physical_activity_minutes'] / 60.0) / (df_fe['daily_screen_time_hours'] + epsilon)
    
    # Interação entre features que podem potenciar o stress
    df_fe['interaction_usage_fatigue'] = df_fe['daily_screen_time_hours'] * df_fe['mental_fatigue_score']
    
    # Variáveis Binárias
    df_fe['dorme_menos_6h'] = np.where(df_fe['sleep_duration_hours'] < 6, 1, 0)
    df_fe['sedentario'] = np.where(df_fe['physical_activity_minutes'] < 30, 1, 0)
    
    # Preparação para cálculo de importância de Features
    # Usaremos o target binário para calcular a feature_importance no modelo inicial
    temp_target = np.where(df_fe['stress_level'] < 5.0, 0, 1)
    drop_cols = ['user_id', 'stress_level']
    X_tmp = df_fe.drop(columns=[c for c in drop_cols if c in df_fe.columns])
    
    # One-Hot Encoding rápido das nominais ('gender', 'occupation')
    categorical_cols = ['gender', 'occupation']
    X_enc = pd.get_dummies(X_tmp, columns=[c for c in categorical_cols if c in X_tmp.columns], drop_first=True)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_enc, temp_target)
    
    importances = pd.Series(rf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
    print("Importância Preliminar das Features (Top 5):")
    print(importances.head())
    
    # Filtro de Correlação (>0.95)
    corr_matrix = X_enc.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"Features removidas devido a correlação cruzada alta: {to_drop}")
    X_enc = X_enc.drop(columns=to_drop)
    
    return X_enc, df_fe

# 1.6 Pré-processamento Final e Divisão dos Dados
def preprocess_and_split(X, y):
    print("\n--- 1.6 Pré-processamento e Divisão dos Dados ---")
    
    # Divisão: 70% treino, 15% validação, 15% teste
    # Primeiro separar 15% de teste garantindo estratificação
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # Para obter 15% do original a partir dos 85% restantes -> test_size = 15/85 = ~0.17647
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), stratify=y_temp, random_state=42)
    
    print(f"Split - Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
    
    # 🔴 Importante: Data Leakage prevention
    # O StandardScaler é "fit" APENAS sobre o conjunto de Treino.
    # Em seguida aplicamos o transform a validação e teste com a matriz de parâmetros do treino.
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Normalização com StandardScaler aplicada APENAS com fit_transform no Treino.")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

# 1.7 Estratégias para Desequilíbrio de Classes
def apply_imbalance_strategies(X_train, y_train):
    print("\n--- 1.7 Estratégias para Desequilíbrio de Classes (Treino) ---")
    if SMOTE is None:
        print("Módulos imbalanced-learn não configurados. Pular etapa.")
        return
        
    print("Contagem Original de Classes:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # 1. SMOTE
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print("\n[SMOTE] Contagem de Classes:")
    print(pd.Series(y_sm).value_counts().sort_index())
    
    # 2. ADASYN
    adasyn = ADASYN(random_state=42)
    X_ad, y_ad = adasyn.fit_resample(X_train, y_train)
    print("\n[ADASYN] Contagem de Classes:")
    print(pd.Series(y_ad).value_counts().sort_index())
    
    # 3. RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    print("\n[RandomUnderSampler] Contagem de Classes:")
    print(pd.Series(y_rus).value_counts().sort_index())
    
    # 4. Exemplo de modelagem com class_weight='balanced'
    print("\n[Nota]: Outra alternativa em algoritmos como SVM/RF é não gerar dados")
    print("mas sim instanciar o modelo com class_weight='balanced'.")

def main():
    print("==================================================")
    print("Execução da Pipeline: 1.4 a 1.7")
    print("==================================================")
    
    filepath = 'sleep_mobile_stress_dataset_15000.csv'
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Erro: O ficheiro '{filepath}' não foi encontrado na raiz deste projeto.")
        return
        
    # 1.4 EDA
    run_eda(df)
    
    # 1.5 Feature Engineering
    X_features, df_fe = feature_engineering(df)
    
    # Converter Target real para usar nos scripts abaixo. 
    # Usaremos Cenário B (5 classes) para mostrar a estratificação de classes complexa.
    df_targets = create_target_scenarios(df)
    y_cenario_B = df_targets['target_B'].astype(int)
    
    # 1.6 Pré-processamento
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = preprocess_and_split(X_features, y_cenario_B)
    
    # 1.7 Tratamento de Desequilíbrio
    apply_imbalance_strategies(X_train_s, y_train)
    
    print("\nPipeline base concluída com sucesso!!!")

if __name__ == '__main__':
    main()
