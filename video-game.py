# Importa a biblioteca pandas, usada para análise e manipulação de dados.
import pandas as pd

# Importa a biblioteca gdown, utilizada para baixar arquivos diretamente do Google Drive.
import gdown

# Define a URL de download do arquivo no Google Drive.
# Essa URL já está formatada com 'uc?export=download&id=' para permitir o download direto.
url = 'https://drive.google.com/uc?export=download&id=1K1jNjCPG5jY-uTuBYPV1nxlKU5sGaah3'

# Define o nome do arquivo de saída que será salvo localmente após o download.
output = 'vgsales.csv'

# Faz o download do arquivo da URL especificada e o salva com o nome definido na variável `output`.
# O terceiro argumento "False" indica que o progresso do download não será exibido.
gdown.download(url, output, False)

# Lê o arquivo CSV baixado e armazena o conteúdo em um DataFrame chamado `df`.
df = pd.read_csv(output)

# Exibe as primeiras 5 linhas do DataFrame para visualização inicial dos dados.
df.head()

# Verifica se há valores ausentes (NaN) em cada linha.
print("Linhas que contém dados ausentes: ", df.isnull().any(axis=1).sum())

# Verifica a quantidade de valores ausentes em cada coluna do DataFrame.
print("Colunas que contém dados ausentes:\n", df.isnull().sum())

# Verifica quantas linhas duplicadas existem no DataFrame.
print("Linhas contendo dados duplicados: ", df.duplicated().sum())

# Remove todas as linhas que contêm ao menos um valor ausente (NaN) do DataFrame.
df = df.dropna()

# Exibe as novas dimensões (número de linhas e colunas) do DataFrame após a exclusão de linhas com dados ausentes.
print("✅ Depois da exclusão:", df.shape)

# Mostra novamente a contagem de valores ausentes por coluna, agora que as linhas com valores ausentes foram removidas.
# Espera-se que o resultado seja zero para todas as colunas.
print("\n🔍 Valores ausentes após remoção:\n", df.isnull().sum())

# Importa o matplotlib para visualizações gráficas.
import matplotlib.pyplot as plt

# Importa o seaborn, uma biblioteca de visualização baseada no matplotlib com estilo mais sofisticado.
import seaborn as sns

# Define o estilo visual dos gráficos do seaborn como "whitegrid" (fundo branco com grades).
sns.set(style="whitegrid")

# ------------------------- Gráfico 1: Top 10 Plataformas -------------------------

# Cria uma figura com tamanho 12x5 polegadas.
plt.figure(figsize=(12, 5))

# Cria um gráfico de contagem (barras) com os 10 principais valores da coluna 'Platform'.
# Garante que as plataformas mais frequentes apareçam primeiro.
sns.countplot(data=df, x='Platform', order=df['Platform'].value_counts().index[:10])

# Define o título do gráfico.
plt.title('Top 10 Plataformas mais comuns')

# Rotaciona os rótulos do eixo X em 45 graus para melhorar a leitura.
plt.xticks(rotation=45)

# Exibe o gráfico.
plt.show()

# Imprime uma linha separadora no console para organização visual.
print('\n' + '-'*140 + '\n')

# ------------------------- Gráfico 2: Distribuição de Gêneros -------------------------

# Cria uma figura com tamanho 10x5 polegadas.
plt.figure(figsize=(10, 5))

# Cria um gráfico de contagem para os gêneros de jogos, ordenando do mais comum para o menos comum.
sns.countplot(data=df, x='Genre', order=df['Genre'].value_counts().index)

# Define o título do gráfico.
plt.title('Distribuição dos Gêneros de Jogos')

# Rotaciona os rótulos do eixo X em 45 graus.
plt.xticks(rotation=45)

# Exibe o gráfico.
plt.show()

# Outra linha separadora para organização visual.
print('\n' + '-'*140 + '\n')

# ------------------------- Gráfico 3: Distribuição por Ano -------------------------

# Cria uma nova figura com tamanho 12x5 polegadas.
plt.figure(figsize=(12, 5))

# Cria um histograma da coluna 'Year', com 30 faixas (bins), sem a curva de densidade (kde=False).
sns.histplot(df['Year'], bins=30, kde=False)

# Define o título e os rótulos dos eixos X e Y.
plt.title('Distribuição de Jogos por Ano')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Número de Jogos')

# Exibe o gráfico.
plt.show()

# Linha separadora no console.
print('\n' + '-'*140 + '\n')

# ------------------------- Gráfico 4: Vendas por Região (Boxplot) -------------------------

# Cria uma nova figura com tamanho 12x6 polegadas.
plt.figure(figsize=(12, 6))

# Define as colunas referentes às vendas por região.
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Transforma o DataFrame de formato largo para longo com `melt`, criando colunas 'Região' e 'Vendas'.
# Isso é necessário para fazer um boxplot comparativo entre as regiões.
df_melted = df[sales_columns].melt(var_name='Região', value_name='Vendas')

# Cria um boxplot que mostra a distribuição das vendas em cada região.
sns.boxplot(x='Região', y='Vendas', data=df_melted)

# Define o título do gráfico.
plt.title('Distribuição das Vendas por Região')

# Exibe o gráfico.
plt.show()

# Importa a biblioteca matplotlib para visualizações.
import matplotlib.pyplot as plt

# Importa a biblioteca seaborn, que fornece gráficos estatísticos com visual mais moderno.
import seaborn as sns

# ------------------------- Boxplot: NA_Sales -------------------------

# Cria uma nova figura com tamanho 10x5 polegadas.
plt.figure(figsize=(10, 5))

# Cria um boxplot para a coluna 'NA_Sales' (vendas na América do Norte).
# Um boxplot mostra a mediana, quartis e possíveis outliers.
sns.boxplot(data=df, y='NA_Sales')

# Define o título do gráfico.
plt.title('Outliers em NA_Sales')

# Exibe o gráfico na tela.
plt.show()

# Imprime uma linha separadora no terminal para melhor organização visual.
print('\n' + '-'*140 + '\n')

# ------------------------- Boxplot: JP_Sales -------------------------

# Cria uma nova figura com tamanho 10x5 polegadas.
plt.figure(figsize=(10, 5))

# Cria um boxplot para a coluna 'JP_Sales' (vendas no Japão).
sns.boxplot(data=df, y='JP_Sales')

# Define o título do gráfico.
plt.title('Outliers em JP_Sales')

# Exibe o gráfico.
plt.show()

# Linha separadora no terminal.
print('\n' + '-'*140 + '\n')

# ------------------------- Boxplot: EU_Sales -------------------------

# Importa bibliotecas, caso ainda não tenha importado
import matplotlib.pyplot as plt
import seaborn as sns

# Cria uma nova figura com tamanho 10x5 polegadas.
plt.figure(figsize=(10, 5))

# Cria o boxplot para a coluna 'EU_Sales' (vendas na Europa).
sns.boxplot(data=df, y='EU_Sales')

# Define o título do gráfico.
plt.title('Outliers em EU_Sales')

# Exibe o gráfico na tela.
plt.show()

print('\n' + '-'*140 + '\n')

# ------------------------- Boxplot: Global_Sales -------------------------

# Cria uma nova figura com tamanho 10x5 polegadas.
plt.figure(figsize=(10, 5))

# Cria um boxplot para a coluna 'Global_Sales' (vendas globais).
sns.boxplot(data=df, y='Global_Sales')

# Define o título do gráfico.
plt.title('Outliers em Global_Sales')

# Exibe o gráfico.
plt.show()

# Importa a biblioteca pandas para manipulação de dados tabulares.
import pandas as pd

# Define a lista de colunas de vendas que serão analisadas para remoção de outliers.
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

# ---------------------- Função: get_iqr_limits ----------------------
# Esta função calcula os limites inferior e superior com base no Intervalo Interquartil
# para detectar outliers em uma coluna numérica.

def get_iqr_limits(df, col):
    # Primeiro quartil (25% dos dados abaixo deste valor)
    Q1 = df[col].quantile(0.25)
    # Terceiro quartil (75% dos dados abaixo deste valor)
    Q3 = df[col].quantile(0.75)
    # Intervalo interquartil (IQR) = Q3 - Q1
    IQR = Q3 - Q1
    # Limite inferior: valores abaixo disso são considerados outliers
    lim_inf = Q1 - 1.5 * IQR
    # Limite superior: valores acima disso são considerados outliers
    lim_sup = Q3 + 1.5 * IQR
    # Retorna os limites calculados
    return lim_inf, lim_sup

# ---------------------- Função: remover_outliers ----------------------
# Esta função aplica a remoção de outliers com base no IQR para múltiplas colunas.

def remover_outliers(df, cols):
    # Cria uma máscara booleana inicialmente verdadeira para todas as linhas.
    mask = pd.Series(True, index=df.index)
    # Para cada coluna fornecida...
    for col in cols:
        # Obtém os limites inferior e superior para a coluna atual.
        lim_inf, lim_sup = get_iqr_limits(df, col)
        # Atualiza a máscara para manter apenas os valores dentro dos limites definidos.
        mask &= df[col].between(lim_inf, lim_sup)
    # Retorna o DataFrame somente com as linhas que passaram pela máscara (sem outliers).
    return df[mask]

# Aplica a função de remoção de outliers ao DataFrame original, considerando as colunas de vendas.
df_sem_outliers = remover_outliers(df, sales_cols)

# Exibe o número de linhas e colunas do DataFrame original.
print("Tamanho original:", df.shape)

# Exibe o número de linhas e colunas após a remoção de outliers.
print("Tamanho após remoção de outliers:", df_sem_outliers.shape)


# Importa o seaborn para visualização estatística dos dados.
import seaborn as sns

# Importa o matplotlib para criação de gráficos.
import matplotlib.pyplot as plt

# Cria uma nova figura com tamanho 12x6 polegadas.
plt.figure(figsize=(12, 6))

# ---------------------- Laço para plotar múltiplos boxplots ----------------------

# Percorre cada coluna de vendas na lista sales_cols com seu índice correspondente.
for i, col in enumerate(sales_cols):
    # Cria um subplot (1 linha, 5 colunas, posição i+1).
    plt.subplot(1, 5, i+1)

    # Cria um boxplot para a coluna atual do DataFrame `df_sem_outliers`.
    # O parâmetro `showfliers=False` oculta os outliers visuais do gráfico.
    sns.boxplot(data=df_sem_outliers, y=col, showfliers=False)

    # Define o título do subplot com o nome da coluna.
    plt.title(col)

# Define um título geral para a figura inteira.
plt.suptitle('📊 Vendas sem Outliers Visuais (showfliers=False)', fontsize=14)

# Ajusta automaticamente o espaçamento entre os subplots para evitar sobreposição.
plt.tight_layout()

# Exibe todos os gráficos.
plt.show()

# Importa a função train_test_split do scikit-learn, usada para dividir os dados em conjuntos de treino e teste.
from sklearn.model_selection import train_test_split

# Cria a matriz de atributos (X) removendo a coluna 'Genre', que é a variável alvo.
X = df_sem_outliers.drop(columns=['Genre'])

# Cria o vetor de rótulos (y), que contém apenas a coluna 'Genre', ou seja, o que queremos prever.
y = df_sem_outliers['Genre']

# Divide os dados em conjunto de treino (80%) e teste (20%).
# - test_size=0.2: 20% dos dados vão para o conjunto de teste.
# - random_state=42: garante reprodutibilidade da divisão.
# - stratify=y: garante que a distribuição das classes (gêneros) em y seja proporcional nos conjuntos de treino e teste.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Exibe o número total de registros no DataFrame após remoção de outliers.
print("Tamanho total:", len(df_sem_outliers))

# Exibe a quantidade de registros no conjunto de treinamento.
print("Treinamento:", len(X_train))

# Exibe a quantidade de registros no conjunto de teste.
print("Teste:", len(X_test))

from sklearn.preprocessing import LabelEncoder

# Função para agrupar plataformas por fabricantes/tipo
def agrupar_plataforma(p):
    if p in ['PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV']:
        return 'PlayStation'
    elif p in ['X360', 'XOne', 'XB']:
        return 'Xbox'
    elif p in ['Wii', 'WiiU', 'GC', 'N64', 'NS']:
        return 'Nintendo Console'
    elif p in ['DS', '3DS', 'GBA', 'GB']:
        return 'Nintendo Portátil'
    else:
        return 'PC/Outros'

# Copia dataframe sem outliers
df = df_sem_outliers.copy()

# executa função de agrupamento de plataformas e coloca na coluna PlataformaAgrupada criada no dataframe
df['Plataform'] = df['Platform'].apply(agrupar_plataforma)

# Criar coluna com a região que teve mais vendas
def regiao_mais_forte(row):
    regioes = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    return max(regioes, key=lambda x: row[x])

df['Top_Region'] = df.apply(regiao_mais_forte, axis=1)

# Remover colunas desnecessárias
df = df.drop(columns=['Name', 'Publisher', 'Global_Sales',
                            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])

# Remover linhas com valores ausentes
df = df.dropna()

# Codificar variáveis categóricas
df['Platform'] = LabelEncoder().fit_transform(df['Platform'])
df['Genre'] = LabelEncoder().fit_transform(df['Genre'])

# Selecionar apenas Platform e Genre como preditores
X = df[['Platform', 'Genre']]
y = df['Top_Region']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

with mlflow.start_run(run_name="RandomForest"):
    # Modelo Random Forest com hiperparâmetros definidos
    clf = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        min_samples_split=15,
        min_samples_leaf=6,
        random_state=42
    )

    # Treinamento
    clf.fit(X_train, y_train)

    # Previsões
    y_pred = clf.predict(X_test)

    # Avaliação
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Logging no MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 5)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("min_samples_split", 15)
    mlflow.log_param("min_samples_leaf", 6)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "modelo_random_forest")

    # Saída
    print("🌲 Acurácia:", acc)
    print("\n📋 Relatório de Classificação:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

with mlflow.start_run(run_name="GradientBoosting"):
    # Dicionário de pesos para balanceamento
    class_weight_person = {
        'NA_Sales': 1.0,
        'EU_Sales': 3.2,
        'JP_Sales': 2.8,
    }

    # Cálculo dos pesos de amostra
    sample_weights = compute_sample_weight(class_weight=class_weight_person, y=y_train)

    # Criação e treino do modelo
    modelo_gb = GradientBoostingClassifier(random_state=42)
    modelo_gb.fit(X_train, y_train, sample_weight=sample_weights)

    # Previsões
    y_pred = modelo_gb.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Logs no MLflow
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(modelo_gb, "modelo_gradient_boosting")

    print("🌟 Resultados do Gradient Boosting Classifier:")
    print("Acurácia:", acc)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

with mlflow.start_run(run_name="KNN"):
    # Instancia e treina o modelo
    modelo_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    modelo_knn.fit(X_train, y_train)

    # Faz previsões
    y_pred_knn = modelo_knn.predict(X_test)

    # Calcula métricas
    acc = accuracy_score(y_test, y_pred_knn)
    report = classification_report(y_test, y_pred_knn, output_dict=True)

    # Loga parâmetros e métricas no MLflow
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("weights", "distance")
    mlflow.log_metric("accuracy", acc)

    # Loga o modelo treinado
    mlflow.sklearn.log_model(modelo_knn, "modelo_knn")

    print("🔍 Resultados do K-Nearest Neighbors Classifier:")
    print("Acurácia:", acc)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_knn))

import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Inicia uma run no MLflow
with mlflow.start_run(run_name="SVM_Model"):

    # Instancia o modelo SVM
    svm = SVC(kernel='linear', C=1.0, random_state=42)

    # Treina o modelo
    svm.fit(X_train, y_train)

    # Faz previsões
    y_pred = svm.predict(X_test)

    # Calcula a acurácia
    acc = accuracy_score(y_test, y_pred)
    print("Acurácia do SVM:", acc)

    # Loga os parâmetros e a métrica no MLflow
    mlflow.log_param("kernel", "linear")
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", acc)

    # Loga o modelo treinado
    mlflow.sklearn.log_model(svm, "svm_model")

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Exibir a matriz de confusão CLF
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test,
    cmap='viridis',  # ou qualquer outro colormap
    display_labels=clf.classes_
)

plt.title("Matriz de Confusão - Random Forest Classifier")
plt.show()

# Exibir a matriz de confusão GB
disp = ConfusionMatrixDisplay.from_estimator(
    modelo_gb, X_test, y_test,
    cmap='viridis',  # ou qualquer outro colormap
    display_labels=modelo_gb.classes_
)

plt.title("Matriz de Confusão - Gradient Boosting")
plt.show()

# Exibir a matriz de confusão k-Nearest Neighbors
disp = ConfusionMatrixDisplay.from_estimator(
    modelo_knn, X_test, y_test,
    cmap='viridis',  # ou qualquer outro colormap
    display_labels=modelo_knn.classes_
)

plt.title("Matriz de Confusão - k-Nearest Neighbors")
plt.show()

# Remover a classe 'Other_Sales' (pouco representativa e atrapalha)
df = df[df['Top_Region'] != 'Other_Sales']

# Tratar a coluna 'Year'
df.loc[:, 'Year'] = pd.to_numeric(df['Year'], errors='coerce')

df = df.dropna(subset=['Year'])

# Remover valores ausentes restantes
df = df.dropna()

# Definir variáveis preditoras e alvo
X = df[['Platform', 'Genre', 'Year']]
y = df['Top_Region']

# Dividir conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [None, 'sqrt']
}

mlflow.set_experiment("GridSearch Random Forest")

with mlflow.start_run(run_name="RandomForest GridSearch"):

    clf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_

    y_pred = best_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log dos melhores hiperparâmetros
    for param, valor in grid_search.best_params_.items():
        mlflow.log_param(param, valor)

    # Log das métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
    mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

    # Log do modelo
    mlflow.sklearn.log_model(best_clf, "random_forest_gridsearch_model")

    print("\n🔍 Melhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    print("\n🔍 Acurácia após otimização:", acc)
    print("\n📋 Relatório de Classificação:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

with mlflow.start_run(run_name="Random Forest Classifier"):
    # Hiperparâmetros
    n_estimators = 200
    max_depth = 10
    min_samples_split = 2
    min_samples_leaf = 2
    random_state = 42

    # Modelo
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    # Treinamento
    clf.fit(X_train, y_train)

    # Predição
    y_pred = clf.predict(X_test)

    # Avaliação
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    # Métricas
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("random_state", random_state)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("cv_accuracy_mean", scores.mean())

    # Modelo
    mlflow.sklearn.log_model(clf, "random_forest_model")

    # Prints opcionais
    print("🔍 Acurácia:", acc)
    print("\n📋 Relatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

class_weight_person = {
    'NA_Sales': 1.0,
    'EU_Sales': 3.2,
    'JP_Sales': 2.8,
}

# Pesos para balanceamento
sample_weights = compute_sample_weight(class_weight=class_weight_person, y=y_train)

param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 3],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', None]
}

mlflow.set_experiment("GridSearch Gradient Boosting")

with mlflow.start_run(run_name="GradientBoosting GridSearch"):

    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    modelo_gb = grid_search.best_estimator_

    y_pred = modelo_gb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log dos melhores parâmetros
    for param, valor in grid_search.best_params_.items():
        mlflow.log_param(param, valor)

    # Log das métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
    mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

    # Log do modelo
    mlflow.sklearn.log_model(modelo_gb, "gradient_boosting_gridsearch_model")

    print("Melhores parâmetros:", grid_search.best_params_)
    print("Acurácia:", acc)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

# Início do experimento
mlflow.set_experiment("Experimento - Gradient Boosting")

with mlflow.start_run(run_name="Gradient Boosting - v1"):
    # Hiperparâmetros
    params = {
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 3,
        'n_estimators': 200,
        'subsample': 0.08,
        'max_features': None,
        'random_state': 42
    }

    modelo_gb = GradientBoostingClassifier(**params)

    # Class weights personalizados
    class_weight_person = {
        'NA_Sales': 1.0,
        'EU_Sales': 3.2,
        'JP_Sales': 2.8,
    }

    # Pesos para balanceamento
    sample_weights = compute_sample_weight(class_weight=class_weight_person, y=y_train)

    # Treinamento
    modelo_gb.fit(X_train, y_train, sample_weight=sample_weights)

    # Avaliação
    y_pred = modelo_gb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Validação cruzada
    scores = cross_val_score(modelo_gb, X, y, cv=5, scoring='accuracy')
    cv_acc = scores.mean()

    # Logging dos hiperparâmetros
    mlflow.log_params(params)

    # Logging das métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("cv_accuracy", cv_acc)
    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
    mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

    # Logging do modelo
    mlflow.sklearn.log_model(modelo_gb, "modelo_gradient_boosting")

    print("Acurácia:", acc)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(modelo_gb, X_test, y_test)

import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_experiment("GridSearch KNN")

with mlflow.start_run(run_name="KNN GridSearch"):

    # Grid de hiperparâmetros
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': [1, 2]
    }

    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    # Treinamento
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_

    y_pred = best_knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log dos melhores hiperparâmetros
    for param, valor in grid_search.best_params_.items():
        mlflow.log_param(param, valor)

    # Log das métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
    mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

    # Log do modelo
    mlflow.sklearn.log_model(best_knn, "knn_gridsearch_model")

    # Print resultados
    print("\n🔍 Melhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)
    print("\n🔍 Acurácia após otimização:", acc)
    print("\n📋 Relatório de Classificação:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score

with mlflow.start_run(run_name="Modelo KNN"):

    # Definir o modelo com hiperparâmetros
    knn = KNeighborsClassifier(
        metric='euclidean',
        n_neighbors=11,
        p=1,
        weights='uniform'
    )

    # Treinar o modelo
    knn.fit(X_train, y_train)

    # Previsões
    y_pred = knn.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Log de parâmetros
    mlflow.log_param("metric", 'euclidean')
    mlflow.log_param("n_neighbors", 11)
    mlflow.log_param("p", 1)
    mlflow.log_param("weights", 'uniform')

    # Log de métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("precision_macro", report['macro avg']['precision'])
    mlflow.log_metric("recall_macro", report['macro avg']['recall'])
    mlflow.log_metric("f1_macro", report['macro avg']['f1-score'])

    # Log do modelo
    mlflow.sklearn.log_model(knn, "knn_model")

# Exibir os resultados
print("🔍 Acurácia:", acc)
print("\n📋 Relatório de Classificação:\n")
print(classification_report(y_test, y_pred, zero_division=0))
print("CV Acurácia média:", cv_mean)

import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Dicionário com pesos personalizados para as classes
class_weight_person = {
    'NA_Sales': 1.0,
    'EU_Sales': 3.2,
    'JP_Sales': 2.8,
}

# Calcula os pesos de amostra para balanceamento
sample_weights = compute_sample_weight(class_weight=class_weight_person, y=y_train)

# Parâmetros do Grid Search
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

mlflow.set_experiment("GridSearch SVM com pesos")

with mlflow.start_run(run_name="SVM GridSearch Pesos"):

    # Configura o Grid Search com SVM
    grid_search = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    # Treina com pesos
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    # Melhor modelo encontrado
    modelo_svm = grid_search.best_estimator_

    # Predição
    y_pred = modelo_svm.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log dos melhores parâmetros do Grid Search
    for param, valor in grid_search.best_params_.items():
        mlflow.log_param(param, valor)

    # Log das métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
    mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

    # Log do modelo
    mlflow.sklearn.log_model(modelo_svm, "svm_gridsearch_pesos")

    # Exibe resultados
    print("Melhores parâmetros:", grid_search.best_params_)
    print("Acurácia:", acc)
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Definindo hiperparâmetros
C = 10
kernel = 'rbf'
gamma = 'scale'
class_weight = 'balanced'
probability = True
random_state = 42

# Inicia experimento MLFlow
mlflow.set_experiment("Experimento - SVM")

with mlflow.start_run(run_name="SVM"):

    # Cria o modelo SVM
    clf = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state
    )

    # Treina
    clf.fit(X_train, y_train)

    # Prediz
    y_pred = clf.predict(X_test)

    # Avalia
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    cv_mean = scores.mean()

    # Log parâmetros
    mlflow.log_param("C", C)
    mlflow.log_param("kernel", kernel)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("probability", probability)

    # Log métricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("cv_mean_accuracy", cv_mean)
    mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
    mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

    # Log do modelo
    mlflow.sklearn.log_model(clf, "svm_model")

    # Imprime resultados
    print("🔍 Acurácia:", acc)
    print("\n📋 Relatório de Classificação:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("✅ CV Acurácia média:", cv_mean)
