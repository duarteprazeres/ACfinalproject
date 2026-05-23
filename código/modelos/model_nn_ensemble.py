"""
model_nn_ensemble.py — Ensemble de Redes Neuronais (PyTorch)
=============================================================
Implementa um ensemble de 5 redes MLP treinadas com PyTorch para os 3 cenários.

Vantagens face ao MLPClassifier do scikit-learn:
  - Dropout real entre camadas (reduz overfitting de forma mais eficaz que L2 isolado)
  - Controlo total sobre a arquitetura e o ciclo de treino
  - Ensemble por média de probabilidades softmax (reduz variância das previsões)

Arquitectura de cada rede:
  Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Output(N)

Referência: Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks
            from Overfitting. JMLR 15, 1929-1958.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from pathlib import Path

from código.modelos.utils import (
    compute_metrics, log_experiment, plot_confusion_matrix_fig,
    split_scenario, save_model, update_registry,
    RANDOM_STATE, GRAFICOS_DIR
)
from código.data_pipeline import load_and_inspect_data, generate_targets, feature_engineering

# =============================================================
# CONFIGURAÇÃO
# =============================================================
EPOCHS     = 300
PATIENCE   = 20
BATCH_SIZE = 32

# Configurações diversas para o ensemble (arquitetura + LR + dropout)
ENSEMBLE_CONFIGS = [
    {'hidden': (128, 64),      'lr': 1e-3,  'dropout': 0.1, 'seed': 42},
    {'hidden': (256, 128, 64), 'lr': 1e-3,  'dropout': 0.1, 'seed': 43},
    {'hidden': (128, 64),      'lr': 5e-4,  'dropout': 0.2, 'seed': 44},
    {'hidden': (64, 32),       'lr': 1e-3,  'dropout': 0.1, 'seed': 45},
    {'hidden': (256, 128),     'lr': 1e-3,  'dropout': 0.1, 'seed': 46},
]
N_MODELS = len(ENSEMBLE_CONFIGS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================
# ARQUITECTURA
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


# =============================================================
# TREINO DE UMA REDE INDIVIDUAL
# =============================================================
def _train_single(X_train, y_train, X_val, y_val, n_classes, cfg):
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(X_train.shape[1], n_classes,
                hidden=cfg['hidden'], dropout=cfg['dropout']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    classes = np.arange(n_classes)
    cw = compute_class_weight('balanced', classes=classes, y=np.array(y_train))
    criterion_train = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))
    criterion_val   = nn.CrossEntropyLoss()  # sem pesos para o early stopping

    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.LongTensor(np.array(y_train)).to(device)
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(np.array(y_val)).to(device)

    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    for epoch in range(EPOCHS):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion_train(model(X_b), y_b).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion_val(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# =============================================================
# INFERÊNCIA DO ENSEMBLE
# =============================================================
def _ensemble_predict(models, X):
    X_t = torch.FloatTensor(X).to(device)
    probas = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(X_t), dim=1).cpu().numpy()
            probas.append(p)
    avg = np.mean(probas, axis=0)
    return avg.argmax(axis=1), avg


# =============================================================
# TREINO DO ENSEMBLE
# =============================================================
def train_nn_ensemble(X_train, y_train, X_val, y_val, scenario):
    print(f"\n  [NN Ensemble PyTorch] Cenário {scenario} — {N_MODELS} redes")
    n_classes = len(np.unique(y_train))
    models = []

    for i, cfg in enumerate(ENSEMBLE_CONFIGS):
        m = _train_single(X_train, y_train, X_val, y_val, n_classes, cfg)
        models.append(m)
        print(f"   Rede {i+1}/{N_MODELS} concluída "
              f"(hidden={cfg['hidden']}, lr={cfg['lr']}, dropout={cfg['dropout']})")

    y_pred, y_proba = _ensemble_predict(models, X_val)
    metrics = compute_metrics(y_val, y_pred, y_proba, scenario)

    print(f"   Ensemble Val F1 Macro: {metrics['f1_macro']:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    log_experiment('NN_Ensemble', scenario,
                   {'n_models': N_MODELS, 'configs': str(ENSEMBLE_CONFIGS),
                    'patience': PATIENCE},
                   metrics, split='val_tuned')
    plot_confusion_matrix_fig(y_val, y_pred, 'NN_Ensemble', scenario)
    return models, metrics


# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 60)
    print("   MODEL_NN_ENSEMBLE.PY — Ensemble PyTorch (5 redes MLP)")
    print("=" * 60)
    print(f"   Device: {device}")

    filepath = 'dataset/sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        filepath = 'sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        print("Erro: ficheiro CSV não encontrado.")
        return

    df           = load_and_inspect_data(filepath)
    df           = generate_targets(df)
    X_raw, df_fe = feature_engineering(df)

    valid = ~df_fe[['target_A', 'target_B_ew', 'target_C_ew']].isnull().any(axis=1)
    X_raw = X_raw[valid].reset_index(drop=True)
    df_fe = df_fe[valid].reset_index(drop=True)

    scenarios = {
        'A': df_fe['target_A'].astype(int),
        'B': df_fe['target_B_ew'].astype(int),
        'C': df_fe['target_C_ew'].astype(int),
    }

    registry = {}
    for sc, y in scenarios.items():
        print(f"\n{'='*60}\n   CENÁRIO {sc}\n{'='*60}")
        X_train, X_val, X_test, y_train, y_val, y_test = split_scenario(
            X_raw.values, y, sc
        )

        # Normalização (fit apenas no treino)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        models, metrics = train_nn_ensemble(X_train, y_train, X_val, y_val, sc)

        # Avaliação no teste
        y_pred_test, y_proba_test = _ensemble_predict(models, X_test)
        metrics_test = compute_metrics(y_test, y_pred_test, y_proba_test, sc)
        print(f"\n   [Teste] F1 Macro: {metrics_test['f1_macro']:.4f}")
        log_experiment('NN_Ensemble', sc,
                       {'n_models': N_MODELS, 'configs': str(ENSEMBLE_CONFIGS)},
                       metrics_test, split='test')

        # Guardar modelos e scaler
        for i, (m, cfg) in enumerate(zip(models, ENSEMBLE_CONFIGS)):
            path = Path('models') / f'nn_ensemble_{sc}_model{i}.pt'
            torch.save({'state_dict': m.state_dict(), 'config': cfg}, path)
        save_model(scaler, f'nn_ensemble_scaler_{sc}.pkl')

        registry[f'nn_ensemble_{sc}'] = {
            'metrics_val':  metrics,
            'metrics_test': metrics_test
        }

    update_registry(registry)
    print("\n✅ model_nn_ensemble.py concluído!")


if __name__ == '__main__':
    main()
