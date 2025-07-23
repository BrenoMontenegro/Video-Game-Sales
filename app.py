from flask import Flask, request, jsonify, render_template
import pandas as pd
from pyngrok import ngrok, conf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Função para treinar modelo
def treinar_modelo(df):
    
    X = df.drop("regiao_mais_forte", axis=1)
    y = df["regiao_mais_forte"]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [None, 'sqrt']
    }

    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # Salvando o modelo
    joblib.dump(best_model, 'modelo.pkl')

    return best_model

# Rota da interface
@app.route('/')
def index():
    return render_template('interface.html')

# Rota de predição (após upload e treino)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'erro': 'Nome do arquivo vazio.'})

    try:
        # Lê o arquivo enviado
        df = pd.read_csv(file)

        # Treina o modelo com os dados enviados
        modelo = treinar_modelo(df)

        # Faz predição 
        X_novos = df.drop("regiao_mais_forte", axis=1)
        predicoes = modelo.predict(X_novos)

        return jsonify({'predicoes': predicoes.tolist()})

    except Exception as e:
        return jsonify({'erro': str(e)})

if __name__ == '__main__':
    conf.get_default().auth_token = "30Cc8cxwu8zYlX5gXdE0JTdQ7jj_7E9mNHWsfeJafbM4nv6Dv"

    public_url = ngrok.connect(5001)
    print("URL pública:", public_url)

    app.run(debug=True)