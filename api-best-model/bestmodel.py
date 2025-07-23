import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

# Carrega o csv
df = pd.read_csv('./vgsales.csv')
# Dropa linhas que tem valor ausente
df = df.dropna()

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
le_genre = LabelEncoder()
df['Genre'] = le_genre.fit_transform(df['Genre'])

le_platform = LabelEncoder()
df['Platform'] = le_platform.fit_transform(df['Platform'])

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