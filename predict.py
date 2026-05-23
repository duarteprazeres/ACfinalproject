"""
predict.py — Previsões com os modelos treinados
================================================
Carrega os modelos guardados em models/ e corre previsões no conjunto de teste
sem voltar a treinar. O conjunto de teste é reproduzível porque usamos o mesmo
RANDOM_STATE=42 e o mesmo split que foi usado no treino.

Uso: python3 predict.py
"""

import os
import sys
import numpy as np
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report

from código.data_pipeline import load_and_inspect_data, generate_targets, feature_engineering
from código.modelos.utils import split_scenario

MODELS_DIR = Path('models')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================
# ARQUITECTURA DO ENSEMBLE (igual à usada no treino)
# =============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_ensemble(scenario, input_dim, n_classes):
    models = []
    for i in range(5):
        path = MODELS_DIR / f'nn_ensemble_{scenario}_model{i}.pt'
        ckpt = torch.load(path, map_location=DEVICE)
        cfg = ckpt['config']
        m = MLP(input_dim, n_classes, hidden=cfg['hidden'], dropout=cfg['dropout'])
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        models.append(m)
    return models


def ensemble_predict(models, X):
    X_t = torch.FloatTensor(X).to(DEVICE)
    probas = []
    for m in models:
        with torch.no_grad():
            p = torch.softmax(m(X_t), dim=1).cpu().numpy()
            probas.append(p)
    avg = np.mean(probas, axis=0)
    return avg.argmax(axis=1)


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    filepath = 'dataset/sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        filepath = 'sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        print("Erro: ficheiro CSV não encontrado.")
        sys.exit(1)

    print("A carregar dados e a preparar conjunto de teste...")
    df = load_and_inspect_data(filepath)
    df = generate_targets(df)
    X_raw, df_fe = feature_engineering(df)

    valid = ~df_fe[['target_A', 'target_B_ew', 'target_C_ew']].isnull().any(axis=1)
    X_raw = X_raw[valid].reset_index(drop=True)
    df_fe = df_fe[valid].reset_index(drop=True)

    scenarios = {
        'A': df_fe['target_A'].astype(int),
        'B': df_fe['target_B_ew'].astype(int),
        'C': df_fe['target_C_ew'].astype(int),
    }

    for sc, y in scenarios.items():
        section(f"CENÁRIO {sc}")
        _, _, X_test, _, _, y_test = split_scenario(X_raw.values, y, sc)
        n_classes = len(np.unique(y))

        # SVM
        svm = joblib.load(MODELS_DIR / f'svm_cenario_{sc}.pkl')
        y_pred_svm = svm.predict(X_test)
        f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
        print(f"\n  SVM          F1 macro = {f1_svm:.4f}")

        # NN/MLP
        nn_model = joblib.load(MODELS_DIR / f'nn_cenario_{sc}.pkl')
        y_pred_nn = nn_model.predict(X_test)
        f1_nn = f1_score(y_test, y_pred_nn, average='macro', zero_division=0)
        print(f"  NN/MLP       F1 macro = {f1_nn:.4f}")

        # XGBoost
        xgb_model = joblib.load(MODELS_DIR / f'xgb_cenario_{sc}.pkl')
        le = joblib.load(MODELS_DIR / f'le_xgb_cenario_{sc}.pkl')
        y_test_enc = le.transform(y_test)
        y_pred_xgb_enc = xgb_model.predict(X_test)
        y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
        f1_xgb = f1_score(y_test, y_pred_xgb, average='macro', zero_division=0)
        print(f"  XGBoost      F1 macro = {f1_xgb:.4f}")

        # Ensemble PyTorch
        scaler = joblib.load(MODELS_DIR / f'nn_ensemble_scaler_{sc}.pkl')
        X_test_scaled = scaler.transform(X_test)
        ensemble_models = load_ensemble(sc, X_test_scaled.shape[1], n_classes)
        y_pred_ens = ensemble_predict(ensemble_models, X_test_scaled)
        f1_ens = f1_score(y_test, y_pred_ens, average='macro', zero_division=0)
        print(f"  Ensemble PT  F1 macro = {f1_ens:.4f}")

    print(f"\n{'=' * 60}")
    print("  Previsões concluídas.")
    print(f"{'=' * 60}\n")
