const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');

const app = express();
const PORT = 4000;

app.use(cors());

// Proxy para o MLflow (ajuste para o IP e porta corretos)
app.use('/mlflow', createProxyMiddleware({
    target: 'http://mlflow_best_model:5001',
    changeOrigin: true,
    pathRewrite: {
        '^/mlflow': '',
    },
}));

app.listen(PORT, () => {
    console.log(`Proxy rodando em http://localhost:${PORT}`);
});