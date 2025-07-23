import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib

def treinar_modelo(df):
    # Pré-processamento
    X = df.drop("target", axis=1)
    y = df["target"]

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
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
        verbose=2,
        scoring='accuracy'
    )

    grid_search.fit(X, y)
    best_clf = grid_search.best_estimator_

    # Salvando o modelo treinado
    joblib.dump(best_clf, "modelo.pkl")

    print("Modelo treinado e salvo com sucesso!")
    print("Melhores hiperparâmetros:", grid_search.best_params_)