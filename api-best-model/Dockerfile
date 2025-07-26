FROM python:3.11-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Instala bibliotecas
RUN pip install --upgrade pip && \
    pip install mlflow scikit-learn pandas numpy

# Expõe a porta do MLflow
EXPOSE 5000

# Comando para servir o modelo salvo com MLflow
CMD mlflow models serve -m /app/mlruns/0/<ID_DO_RUN>/artifacts/MLP_model --host 0.0.0.0 --port 5000
