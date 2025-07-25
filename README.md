# Video-Game-Sales

### Projeto de Machine Learning

### Discentes: Alrykemes Cavalcanti, Breno Montenegro, Elton Alves

### Docente: Antônio Correia de Sá Barreto Neto

<br>

## Descrição:

Foram desenvolvidos 6 modelos de Machine Learning para prever um problema de classificação da base de dados videogame-sales disponível em: https://www.kaggle.com/datasets/gregorut/videogamesales

Este projeto conta com uma Interface onde é possível inserir os valores para predição e receber o resultado do melhor modelo escolhido entre os 6 desenvolvidos

### O problema de classificação escolhido foi:

Prever a melhor região para vender o jogo de acordo com seu gênero e plataforma, com o intuito de aumentar a quantidade de vendas.

### Tecnologias Utilizadas:

-   Python
-   Pandas
-   MatplotLib
-   MlFlow
-   TensorFlow
-   Sklearn
-   Node
-   Nginx
-   Docker

### Instruções para Deploy:

-   Clone o repositório e acesse a pasta do projeto, ao acessar o diretório principal, entre na pasta proxy e execute `npm install`.
-   Depois disto retorne ao diretório principal do projeto e nele execute `docker compose up --build -d`
-   Após isso você pode acessar as seguintes portas: 
    - localhost:3000 -> Webapp com interface e predição. 
    - localhost:4000 -> Proxy reverso utilizado pelo webapp para acessar o mlflow server.
    - localhost:5000 -> Ui do mlflow com detalhes de todos os modelos. 
    - localhost:5001 -> Server do mlflow com API do principal modelo no endpoint `/invocations` para request http deve conter no body um json com as linhas e colunas de treinamento. Ex abaixo: 
        ```json
            {
                "dataframe_split": {
                "columns": ["Platform", "Genre", "Year"],
                "data": [
                    [0, 4, 0],
                    [4, 4, 0]
                ]
                }
            } 

