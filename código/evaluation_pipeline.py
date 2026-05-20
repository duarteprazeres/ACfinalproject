"""
evaluation_pipeline.py — Análise de Erros e Avaliação Final
================================================
3.1 Análise de Erros
3.2 Interpretabilidade (SHAP, Permutation Importance)
3.3 Testes de Significância Estatística
3.4 Avaliação Final no Conjunto de Teste (Hold-out)
3.5 Tabela Comparativa Final
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

from utils import (
    split_scenario, CV_FOLDS, RANDOM_STATE, GRAFICOS_DIR, MODELS_DIR
)
from data_pipeline import load_and_inspect_data, generate_targets, feature_engineering

# ── statsmodels McNemar ──────────────────────────────────────────────────────
try:
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels não disponível – McNemar test ignorado.")

GRAFICOS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SCENARIOS   = ['A', 'B', 'C']
MODEL_NAMES = ['svm', 'nn', 'xgb']

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    """
    Carrega o dataset e prepara os dados utilizando as funções do data_pipeline.
    Retorna as features cruas, o dataframe com as features processadas e os dicionários
    de targets para os cenários A, B e C, removendo eventuais valores nulos introduzidos na transformação.
    """
    filepath = 'dataset/sleep_mobile_stress_dataset_15000.csv'
    if not os.path.exists(filepath):
        filepath = 'sleep_mobile_stress_dataset_15000.csv'
    df           = load_and_inspect_data(filepath)
    df           = generate_targets(df)
    X_raw, df_fe = feature_engineering(df)
    valid        = ~df_fe[['target_A','target_B','target_C']].isnull().any(axis=1)
    X_raw        = X_raw[valid].reset_index(drop=True)
    df_fe        = df_fe[valid].reset_index(drop=True)
    targets = {
        'A': df_fe['target_A'].astype(int),
        'B': df_fe['target_B'].astype(int),
        'C': df_fe['target_C'].astype(int),
    }
    return X_raw, df_fe, targets


# ════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════════════════════
def load_models():
    """
    Carrega os modelos treinados e serializados (.pkl) a partir do diretório predefinido.
    Isto inclui os modelos de SVM, Neural Networks, XGBoost e os respetivos codificadores de labels (LabelEncoders).
    """
    models = {}
    for sc in SCENARIOS:
        models[f'svm_{sc}'] = joblib.load(MODELS_DIR / f'svm_cenario_{sc}.pkl')
        models[f'nn_{sc}']  = joblib.load(MODELS_DIR / f'nn_cenario_{sc}.pkl')
        models[f'xgb_{sc}'] = joblib.load(MODELS_DIR / f'xgb_cenario_{sc}.pkl')
        models[f'le_xgb_{sc}'] = joblib.load(MODELS_DIR / f'le_xgb_cenario_{sc}.pkl')
    print("✅ Modelos carregados.")
    return models


# ════════════════════════════════════════════════════════════════════════════
# 3.1  ANÁLISE DE ERROS
# ════════════════════════════════════════════════════════════════════════════
def section_31(models, X_raw, df_fe, targets):
    """
    Secção 3.1: Análise de Erros.
    Gera matrizes de confusão (contagens e normalizadas), relatórios de classificação (precision, recall, f1)
    e extrai os exemplos classificados incorretamente para ficheiros CSV.
    Adicionalmente, traça curvas de aprendizagem para SVM e XGBoost, e curvas de perda (loss curves) para Redes Neuronais.
    """
    print("\n" + "="*60)
    print("  3.1 — ANÁLISE DE ERROS")
    print("="*60)

    feat_names = X_raw.columns.tolist()

    for sc, y in targets.items():
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_scenario(X_raw.values, y, sc)

        for mname in MODEL_NAMES:
            key = f'{mname}_{sc}'
            mdl = models[key]

            # ── predictions ──────────────────────────────────────────────
            if mname == 'xgb':
                le       = models[f'le_xgb_{sc}']
                yp_enc   = mdl.predict(X_val)
                y_pred   = le.inverse_transform(yp_enc)
                y_proba  = mdl.predict_proba(X_val)
            else:
                y_pred   = mdl.predict(X_val)
                y_proba  = mdl.predict_proba(X_val)

            labels = sorted(y_val.unique())

            # ── confusion matrix (counts) ─────────────────────────────────
            cm   = confusion_matrix(y_val, y_pred, labels=labels)
            n    = len(labels)
            fig, axes = plt.subplots(1, 2, figsize=(max(10, n*2), max(4, n)))
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(
                ax=axes[0], colorbar=False, cmap='Blues')
            axes[0].set_title(f'{mname.upper()} | Cen.{sc} — Counts')
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(
                ax=axes[1], colorbar=True, cmap='Blues', values_format='.2f')
            axes[1].set_title(f'{mname.upper()} | Cen.{sc} — Normalised')
            plt.tight_layout()
            path = GRAFICOS_DIR / f'cm_detail_{mname}_{sc}.png'
            plt.savefig(path, dpi=130); plt.close()
            print(f"   CM saved: {path}")

            # ── classification report CSV ─────────────────────────────────
            rep = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T
            rep_df.to_csv(GRAFICOS_DIR / f'clf_report_{mname}_{sc}.csv')

            # ── error cases ───────────────────────────────────────────────
            err_mask = (np.array(y_val) != y_pred)
            if err_mask.sum() > 0:
                err_df = pd.DataFrame(X_val[err_mask], columns=feat_names)
                err_df['y_true'] = np.array(y_val)[err_mask]
                err_df['y_pred'] = y_pred[err_mask]
                err_df.to_csv(GRAFICOS_DIR / f'errors_{mname}_{sc}.csv', index=False)
                top_conf = (
                    pd.DataFrame({'true': np.array(y_val)[err_mask], 'pred': y_pred[err_mask]})
                    .value_counts().head(5)
                )
                print(f"   [{mname.upper()} {sc}] Top confusions:\n{top_conf.to_string()}")

        # ── learning curves (SVM & XGB only – NN too slow) ───────────────
        for mname in ['svm', 'xgb']:
            key = f'{mname}_{sc}'
            mdl = models[key]
            X_lc = X_raw.values if mname == 'xgb' else X_raw.values
            y_lc = y.copy()
            if mname == 'xgb':
                le  = models[f'le_xgb_{sc}']
                y_lc = le.transform(y_lc)

            cv_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            train_sizes, train_scores, val_scores = learning_curve(
                mdl, X_lc, y_lc,
                train_sizes=np.linspace(0.2, 1.0, 5),
                cv=cv_lc, scoring='f1_macro', n_jobs=-1
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(train_sizes, train_scores.mean(1), 'o-', label='Train F1')
            ax.fill_between(train_sizes,
                            train_scores.mean(1)-train_scores.std(1),
                            train_scores.mean(1)+train_scores.std(1), alpha=.2)
            ax.plot(train_sizes, val_scores.mean(1),   's--', label='Val F1')
            ax.fill_between(train_sizes,
                            val_scores.mean(1)-val_scores.std(1),
                            val_scores.mean(1)+val_scores.std(1), alpha=.2)
            ax.set_xlabel('Training samples'); ax.set_ylabel('F1 Macro')
            ax.set_title(f'Learning Curve — {mname.upper()} | Cenário {sc}')
            ax.legend(); plt.tight_layout()
            path = GRAFICOS_DIR / f'learning_curve_{mname}_{sc}.png'
            plt.savefig(path, dpi=130); plt.close()
            print(f"   LC saved: {path}")

        # ── NN loss curves ────────────────────────────────────────────────
        nn_mdl = models[f'nn_{sc}']
        mlp    = nn_mdl.named_steps['mlp']
        if hasattr(mlp, 'loss_curve_') and mlp.loss_curve_:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(mlp.loss_curve_, label='Train Loss', color='#4C72B0')
            if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
                ax2 = ax.twinx()
                ax2.plot(mlp.validation_scores_, '--', color='#DD8452', label='Val Score')
                ax2.set_ylabel('Val Score')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(f'NN Loss Curve — Cenário {sc}')
            ax.legend(loc='upper right'); plt.tight_layout()
            path = GRAFICOS_DIR / f'nn_loss_phase3_{sc}.png'
            plt.savefig(path, dpi=130); plt.close()
            print(f"   NN loss curve saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 3.2  INTERPRETABILIDADE
# ════════════════════════════════════════════════════════════════════════════
def section_32(models, X_raw, targets):
    """
    Secção 3.2: Interpretabilidade.
    Avalia e visualiza a importância das features para os modelos treinados.
    Utiliza Feature Importance nativa e SHAP para o XGBoost, e Permutation Importance (agnóstica ao modelo)
    para o SVM, Redes Neuronais e XGBoost, permitindo perceber quais as variáveis mais influentes.
    """
    print("\n" + "="*60)
    print("  3.2 — INTERPRETABILIDADE")
    print("="*60)

    feat_names = X_raw.columns.tolist()

    for sc, y in targets.items():
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_scenario(X_raw.values, y, sc)

        # ── XGBoost feature importance ────────────────────────────────────
        xgb_mdl = models[f'xgb_{sc}']
        imp     = pd.Series(xgb_mdl.feature_importances_, index=feat_names).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.head(15).sort_values().plot.barh(ax=ax, color='#4C72B0')
        ax.set_title(f'XGBoost Feature Importance — Cenário {sc}')
        ax.set_xlabel('Importance (gain)')
        plt.tight_layout()
        path = GRAFICOS_DIR / f'feat_imp_xgb_{sc}.png'
        plt.savefig(path, dpi=130); plt.close()
        print(f"   FeatImp saved: {path}")

        # ── SHAP — XGBoost ────────────────────────────────────────────────
        try:
            le_xgb   = models[f'le_xgb_{sc}']
            explainer = shap.TreeExplainer(xgb_mdl)
            shap_vals = explainer.shap_values(X_val[:300])
            if isinstance(shap_vals, list):
                shap_arr = np.abs(np.stack(shap_vals)).mean(0)
            else:
                shap_arr = shap_vals
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_arr, X_val[:300], feature_names=feat_names,
                              show=False, plot_type='bar')
            plt.title(f'SHAP Summary — XGBoost | Cenário {sc}')
            plt.tight_layout()
            path = GRAFICOS_DIR / f'shap_xgb_{sc}.png'
            plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close()
            print(f"   SHAP saved: {path}")
        except Exception as e:
            print(f"   SHAP XGB Cenário {sc} falhou: {e}")

        # ── Permutation Importance — all models ───────────────────────────
        for mname in MODEL_NAMES:
            mdl = models[f'{mname}_{sc}']
            if mname == 'xgb':
                le      = models[f'le_xgb_{sc}']
                y_v_enc = le.transform(y_val)
                pi = permutation_importance(mdl, X_val, y_v_enc,
                                            n_repeats=10, random_state=RANDOM_STATE,
                                            scoring='f1_macro', n_jobs=-1)
            else:
                pi = permutation_importance(mdl, X_val, np.array(y_val),
                                            n_repeats=10, random_state=RANDOM_STATE,
                                            scoring='f1_macro', n_jobs=-1)

            pi_df = pd.DataFrame({
                'feature': feat_names,
                'mean':    pi.importances_mean,
                'std':     pi.importances_std
            }).sort_values('mean', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(pi_df['feature'][::-1], pi_df['mean'][::-1],
                    xerr=pi_df['std'][::-1], color='#DD8452', alpha=0.85)
            ax.set_title(f'Permutation Importance — {mname.upper()} | Cenário {sc}')
            ax.set_xlabel('Mean decrease in F1 Macro')
            plt.tight_layout()
            path = GRAFICOS_DIR / f'perm_imp_{mname}_{sc}.png'
            plt.savefig(path, dpi=130); plt.close()
            print(f"   PermImp saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 3.3  TESTES DE SIGNIFICÂNCIA ESTATÍSTICA
# ════════════════════════════════════════════════════════════════════════════
def section_33(models, X_raw, targets):
    """
    Secção 3.3: Testes de Significância Estatística.
    Realiza o Teste de McNemar (para comparar predições pareadas) e o Teste de Wilcoxon
    (para comparar as pontuações F1 nas dobras da validação cruzada) entre os modelos,
    verificando se a diferença de performance é estatisticamente significativa.
    """
    print("\n" + "="*60)
    print("  3.3 — TESTES DE SIGNIFICÂNCIA ESTATÍSTICA")
    print("="*60)

    pairs   = [('svm','nn'), ('svm','xgb'), ('nn','xgb')]
    results = []

    for sc, y in targets.items():
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_scenario(X_raw.values, y, sc)
        y_val_arr = np.array(y_val)

        # ── predictions per model ─────────────────────────────────────────
        preds = {}
        for mname in MODEL_NAMES:
            mdl = models[f'{mname}_{sc}']
            if mname == 'xgb':
                le         = models[f'le_xgb_{sc}']
                preds[mname] = le.inverse_transform(mdl.predict(X_val))
            else:
                preds[mname] = mdl.predict(X_val)

        # ── McNemar ───────────────────────────────────────────────────────
        for m1, m2 in pairs:
            c1 = (preds[m1] == y_val_arr)
            c2 = (preds[m2] == y_val_arr)
            b  = np.sum( c1 & ~c2)
            c  = np.sum(~c1 &  c2)
            row = {'scenario': sc, 'pair': f'{m1} vs {m2}', 'b': b, 'c': c}
            if HAS_SM and (b + c) > 0:
                ct  = np.array([[np.sum(c1 & c2), b],[c, np.sum(~c1 & ~c2)]])
                res = sm_mcnemar(ct, exact=True)
                row['mcnemar_p'] = round(res.pvalue, 4)
                sig = '★' if res.pvalue < 0.05 else ''
                print(f"   McNemar [{sc}] {m1} vs {m2}: p={res.pvalue:.4f} {sig}")
            else:
                row['mcnemar_p'] = float('nan')
            results.append(row)

    # ── Wilcoxon on CV fold scores ────────────────────────────────────────
    print("\n  Computing CV fold scores for Wilcoxon (SVM & XGB fast, NN reduced)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    wilcoxon_rows = []

    for sc, y in targets.items():
        fold_scores = {}
        for mname in MODEL_NAMES:
            mdl  = models[f'{mname}_{sc}']
            y_cv = y.copy()
            if mname == 'xgb':
                le   = models[f'le_xgb_{sc}']
                y_cv = pd.Series(le.transform(y_cv))
                if mname == 'nn':
                    mdl.set_params(mlp__max_iter=50)
            scores = cross_val_score(mdl, X_raw.values, y_cv,
                                     cv=cv, scoring='f1_macro', n_jobs=-1)
            fold_scores[mname] = scores

        for m1, m2 in pairs:
            diff = fold_scores[m1] - fold_scores[m2]
            if np.all(diff == 0):
                p_val = 1.0
            else:
                try:
                    _, p_val = wilcoxon(fold_scores[m1], fold_scores[m2])
                except Exception:
                    p_val = float('nan')
            sig = '★' if (not np.isnan(p_val) and p_val < 0.05) else ''
            print(f"   Wilcoxon [{sc}] {m1} vs {m2}: p={p_val:.4f} {sig}")
            wilcoxon_rows.append({
                'scenario': sc, 'pair': f'{m1} vs {m2}',
                'wilcoxon_p': round(p_val, 4), 'significant': sig
            })

    # ── save tables ───────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(GRAFICOS_DIR / 'mcnemar_results.csv', index=False)
    pd.DataFrame(wilcoxon_rows).to_csv(GRAFICOS_DIR / 'wilcoxon_results.csv', index=False)
    print("   Statistical test results saved.")
    return pd.DataFrame(results), pd.DataFrame(wilcoxon_rows)


# ════════════════════════════════════════════════════════════════════════════
# 3.4  AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ════════════════════════════════════════════════════════════════════════════
def section_34(models, X_raw, targets):
    """
    Secção 3.4: Avaliação Final no Conjunto de Teste.
    Calcula diversas métricas de desempenho (F1-Macro, F1-Weighted, Accuracy, Cohen's Kappa, MCC, ROC-AUC)
    num conjunto de dados retido (Hold-Out) e exporta os resultados para um ficheiro CSV.
    """
    print("\n" + "="*60)
    print("  3.4 — AVALIAÇÃO FINAL HOLD-OUT (TEST SET)")
    print("="*60)

    rows = []
    for sc, y in targets.items():
        _, _, X_te, _, _, y_te = split_scenario(X_raw.values, y, sc)
        y_te_arr = np.array(y_te)
        n_cls    = len(np.unique(y_te_arr))

        for mname in MODEL_NAMES:
            mdl = models[f'{mname}_{sc}']
            if mname == 'xgb':
                le       = models[f'le_xgb_{sc}']
                yp_enc   = mdl.predict(X_te)
                y_pred   = le.inverse_transform(yp_enc)
                y_proba  = mdl.predict_proba(X_te)
            else:
                y_pred   = mdl.predict(X_te)
                y_proba  = mdl.predict_proba(X_te)

            f1_mac  = f1_score(y_te_arr, y_pred, average='macro',    zero_division=0)
            f1_w    = f1_score(y_te_arr, y_pred, average='weighted', zero_division=0)
            acc     = accuracy_score(y_te_arr, y_pred)
            kappa   = cohen_kappa_score(y_te_arr, y_pred)
            mcc     = matthews_corrcoef(y_te_arr, y_pred)
            try:
                if sc == 'A':
                    auc = roc_auc_score(y_te_arr, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_te_arr, y_proba, multi_class='ovr', average='macro')
            except Exception:
                auc = float('nan')

            top2 = float('nan')
            if sc == 'C':
                try:
                    top2_idx = np.argsort(y_proba, axis=1)[:, -2:]
                    top2 = float(np.mean([y_te_arr[i] in top2_idx[i] for i in range(len(y_te_arr))]))
                except Exception:
                    pass

            row = {
                'model': mname.upper(), 'scenario': sc,
                'f1_macro_test': round(f1_mac, 4),
                'f1_weighted_test': round(f1_w, 4),
                'accuracy_test': round(acc, 4),
                'cohen_kappa_test': round(kappa, 4),
                'mcc_test': round(mcc, 4),
                'roc_auc_test': round(auc, 4) if not np.isnan(auc) else float('nan'),
                'top2_acc_test': round(top2, 4) if not np.isnan(top2) else float('nan'),
            }
            rows.append(row)
            print(f"   {mname.upper()} | {sc}: F1={f1_mac:.4f}  Acc={acc:.4f}  κ={kappa:.4f}  AUC={auc:.4f}")

    test_df = pd.DataFrame(rows)
    test_df.to_csv(GRAFICOS_DIR / 'test_results.csv', index=False)
    print(f"\n   Test results saved → graficos/test_results.csv")
    return test_df


# ════════════════════════════════════════════════════════════════════════════
# 3.5  TABELA COMPARATIVA FINAL
# ════════════════════════════════════════════════════════════════════════════
def section_35(test_df, X_raw, targets, models):
    """
    Secção 3.5: Tabela Comparativa Final.
    Junta a performance média obtida na validação cruzada (CV) com as métricas da avaliação final (Test),
    gerando e guardando uma tabela formatada visualmente (PNG) que resume o desempenho de todos os modelos.
    """
    print("\n" + "="*60)
    print("  3.5 — TABELA COMPARATIVA FINAL")
    print("="*60)

    # CV scores (mean ± std) per model × scenario
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for sc, y in targets.items():
        for mname in MODEL_NAMES:
            mdl  = models[f'{mname}_{sc}']
            y_cv = y.copy()
            if mname == 'xgb':
                le   = models[f'le_xgb_{sc}']
                y_cv = pd.Series(le.transform(y_cv))
            scores = cross_val_score(mdl, X_raw.values, y_cv,
                                     cv=cv, scoring='f1_macro', n_jobs=-1)
            cv_rows.append({
                'model': mname.upper(), 'scenario': sc,
                'f1_macro_cv': f"{scores.mean():.4f} ± {scores.std():.4f}",
                'f1_macro_cv_mean': round(scores.mean(), 4),
            })
    cv_df = pd.DataFrame(cv_rows)

    # Merge with test results
    final = cv_df.merge(test_df, on=['model','scenario'])
    cols  = ['model','scenario','f1_macro_cv',
             'f1_macro_test','accuracy_test','cohen_kappa_test','mcc_test','roc_auc_test']
    final = final[cols].rename(columns={
        'f1_macro_cv':        'F1-macro (CV)',
        'f1_macro_test':      'F1-macro (Test)',
        'accuracy_test':      'Accuracy',
        'cohen_kappa_test':   "Cohen's κ",
        'mcc_test':           'MCC',
        'roc_auc_test':       'ROC-AUC',
    })
    final = final.sort_values(['scenario','model']).reset_index(drop=True)
    final.to_csv(GRAFICOS_DIR / 'tabela_final_resultados.csv', index=False)
    print(final.to_string(index=False))

    # Render as PNG table
    fig, ax = plt.subplots(figsize=(14, 0.45 * (len(final) + 2)))
    ax.axis('off')
    tbl = ax.table(
        cellText=final.values,
        colLabels=final.columns,
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(final.columns))))

    # Header styling
    for j in range(len(final.columns)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colours
    colors = ['#EBF5FB', '#FDFEFE']
    for i in range(1, len(final)+1):
        for j in range(len(final.columns)):
            tbl[i, j].set_facecolor(colors[i % 2])

    plt.title('Tabela Comparativa Final — Fase 3', fontsize=13,
              fontweight='bold', pad=12, color='#2C3E50')
    plt.tight_layout()
    path = GRAFICOS_DIR / 'tabela_final_resultados.png'
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n   Tabela PNG saved → {path}")
    return final


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    """
    Função principal que coordena a execução sequencial de todas as secções da pipeline de avaliação.
    Garante o carregamento de dados e modelos, e invoca as análises de erro, interpretabilidade,
    significância e avaliação final em série.
    """
    print("=" * 60)
    print("   EVALUATION_PIPELINE.PY — Análise de Erros e Avaliação Final")
    print("=" * 60)

    X_raw, df_fe, targets = load_data()
    models                = load_models()

    section_31(models, X_raw, df_fe, targets)
    section_32(models, X_raw, targets)
    mn, wn = section_33(models, X_raw, targets)
    test_df = section_34(models, X_raw, targets)
    final   = section_35(test_df, X_raw, targets, models)

    print("\n" + "="*60)
    print("✅ FASE 3 CONCLUÍDA — todos os gráficos em graficos/")
    print("="*60)


if __name__ == '__main__':
    main()
