from flask import Flask, request, jsonify, render_template
import pandas as pd
from pyngrok import ngrok, conf

# from videogame import predict_dataset  
# Adicionar função no arquivo do modelo (to do)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'erro': 'Nenhum arquivo enviado.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'erro': 'Nome do arquivo vazio.'})

    try:
        df = pd.read_csv(file)
        resultado = predict_dataset(df)
        return jsonify({'predicoes': resultado.tolist()})
    except Exception as e:
        return jsonify({'erro': str(e)})

if __name__ == '__main__':
    conf.get_default().auth_token = "30Cc8cxwu8zYlX5gXdE0JTdQ7jj_7E9mNHWsfeJafbM4nv6Dv"

    public_url = ngrok.connect(5001)
    print("URL pública:", public_url)

    app.run(debug=True)