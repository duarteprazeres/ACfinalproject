
import pandas as pd
import copy


# 1. Carregar o dataset
df_original = pd.read_csv('sleep_mobile_stress_dataset_15000.csv')


# 2. Verificar o tamanho (shape) - linhas e colunas
print("Shape do dataset (linhas, colunas):", df_original.shape)


# 3. Verificar os nomes das colunas e os tipos de dados (dtypes)
print("\nTipos de Dados:")
print(df_original.dtypes)



# 4. Identificar colunas irrelevantes (IDs, datas, etc.) antes das estatísticas
irrelevant_cols = []
for col in df_original.columns:
    if 'id' in col.lower():
        irrelevant_cols.append(col)
if irrelevant_cols:
    print("\nColunas potencialmente irrelevantes:", irrelevant_cols)
else:
    print("\nNenhuma coluna potencialmente irrelevante identificada.")

# 5. Estatísticas descritivas (removendo user_id se existir)
desc_cols = [c for c in df_original.columns if c not in irrelevant_cols]
print("\nEstatísticas Descritivas (sem colunas irrelevantes):")
print(df_original[desc_cols].describe())


# 6. Identificar a variável alvo contínua (stress_level)
target_col = 'stress_level'
if target_col in df_original.columns:
    print(f"\nVariável alvo encontrada: {target_col}")
    print("Tipo:", df_original[target_col].dtype)
    print("Valores únicos:", df_original[target_col].nunique())
    print("Mínimo:", df_original[target_col].min())
    print("Máximo:", df_original[target_col].max())
    print("Distribuição (amostra):")
    print(df_original[target_col].value_counts(bins=10, sort=False))
else:
    print("Variável alvo 'stress_level' não encontrada.")



# Para alterar o dataset sem mexer no original:
df = copy.deepcopy(df_original)

print("\nValores em Falta (Nulos)")
print(df.isnull().sum())

print("\nValores Duplicados")
print(df.duplicated().sum())
df=df.drop_duplicates()

# 1. Definir o ponto de corte (experimentamos com 5.0 e depois o 7.38) -> optamos por 7.38 para ter um equilíbrio melhor entre as classes pois os algoritmos de Machine Learning funcionam muito melhor se tiverem o mesmo número de exemplos (ou distribuição similar) para aprender a classe 0 e a classe 1 (dados balanceados).
limiar_A = 7.38

# 2. Criar a nova coluna usando o numpy (np.where)
# Lê-se: Se stress_level >= limiar, preenche com 1, senão preenche com 0
import numpy as np
df['target_A_binario'] = np.where(df['stress_level'] >= limiar_A, 1, 0)

# 3. Ver quantas pessoas ficaram em cada grupo
print("\nDistribuição Cenário A (0 = Sem Stress, 1 = Com Stress):")
print(df['target_A_binario'].value_counts())

# Cenário B: Multiclasse com 5 níveis (Equal-frequency binning)
# O q=5 diz ao Pandas para criar 5 "baldes" com o mesmo número de pessoas (quantis)
# duplicates='drop' resolve o problema de arestas duplicadas quando há muitos valores repetidos
df['target_B_5classes'] = pd.qcut(df['stress_level'], q=5, duplicates='drop')

# Mapear os valores categóricos para nomes mais descritivos
nomes_classes_B = {cat: nome for cat, nome in zip(df['target_B_5classes'].cat.categories, 
                                                    ['Nenhum', 'Baixo', 'Moderado', 'Alto', 'Extremo'][:len(df['target_B_5classes'].cat.categories)])}
df['target_B_5classes'] = df['target_B_5classes'].map(nomes_classes_B)

print("\nDistribuição Cenário B (5 Classes):")
print(df['target_B_5classes'].value_counts())

from sklearn.cluster import KMeans

# Definir os nomes genéricos para as 10 classes
nomes_10_classes = [f"Nivel_{i}" for i in range(1, 11)]

# --- Método 1: Equal-Width (pd.cut) ---
df['target_C_width'] = pd.cut(df['stress_level'], bins=10, labels=nomes_10_classes)

# --- Método 2: Equal-Frequency (pd.qcut) ---
# Usamos duplicates='drop' porque às vezes os dados repetem-se muito num ponto 
# e o qcut não consegue dividir em partes perfeitamente iguais.
df['target_C_freq'] = pd.qcut(df['stress_level'], q=10, duplicates='drop')

# Renomear as categorias para nomes mais legíveis
n_cats_freq = df['target_C_freq'].cat.categories.size
nomes_freq = [f"Nivel_{i}" for i in range(1, n_cats_freq + 1)]
nomes_dict_freq = {cat: nome for cat, nome in zip(df['target_C_freq'].cat.categories, nomes_freq)}
df['target_C_freq'] = df['target_C_freq'].map(nomes_dict_freq)

# --- Método 3: K-Means Clustering ---
# O KMeans precisa que os dados estejam no formato de matriz 2D (por isso o reshape)
kmeans = KMeans(n_clusters=10, random_state=42)
grupos_kmeans = kmeans.fit_predict(df[['stress_level']])

# O KMeans não garante a ordem das classes (o grupo 0 pode ter stress alto).
# Precisamos de ordenar os grupos para que façam sentido (0 = menor stress).
centros = kmeans.cluster_centers_.flatten()
ordem = centros.argsort()
mapa_ordem = {ordem[i]: f"Nivel_{i+1}" for i in range(10)}

df['target_C_kmeans'] = [mapa_ordem[grupo] for grupo in grupos_kmeans]


# --- Imprimir e Comparar os Resultados ---
print("\n=== COMPARAÇÃO DO CENÁRIO C (10 Classes) ===\n")

comparacao = pd.DataFrame({
    'Equal-Width (cut)': df['target_C_width'].value_counts(),
    'Equal-Freq (qcut)': df['target_C_freq'].value_counts(),
    'K-Means': df['target_C_kmeans'].value_counts()
}).fillna(0).astype(int)

print(comparacao)