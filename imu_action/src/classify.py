"""
classify.py — 分类器训练与 LOOCV 评估

接口:
  train_and_evaluate(features_csv, out_dir) -> dict
  predict_segment(model_path, segment_array, feature_cols) -> (label, prob)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

from config import (
    RANDOM_STATE, RF_N_ESTIMATORS,
    SVM_C_GRID, SVM_GAMMA_GRID, KNN_K,
    TARGET_LOOCV_ACC, TARGET_ROBUST_ACC,
    REPORTS, DATA_SEGMENTS,
)


def _loocv(model_cls, model_kwargs, X: np.ndarray, y: np.ndarray, label_names: list[str]):
    loo = LeaveOneOut()
    preds = []
    trues = []
    probs = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = model_cls(**model_kwargs, random_state=RANDOM_STATE) if "random_state" in model_cls.__init__.__code__.co_varnames else model_cls(**model_kwargs)
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)[0]
        preds.append(pred)
        trues.append(y_test[0])
        if hasattr(clf, "predict_proba"):
            probs.append(clf.predict_proba(X_test_s)[0])
        else:
            probs.append(None)
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds, labels=label_names)
    report = classification_report(trues, preds, labels=label_names, target_names=label_names, zero_division=0)
    return acc, cm, report, trues, preds, probs


def train_and_evaluate(features_df: pd.DataFrame, selected_cols: list[str], out_dir: Path | str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 过滤掉可疑段（可选，这里保留全部但报告）
    # mask = ~features_df["suspicious"] if "suspicious" in features_df.columns else pd.Series([True] * len(features_df))
    # df = features_df[mask].copy()
    # print(f"[Classify] 使用 {len(df)} 段 (过滤掉 {len(features_df) - len(df)} 可疑段)")
    df = features_df.copy()
    print(f"[Classify] 使用全部 {len(df)} 段 (可疑段保留)")

    X = df[selected_cols].values
    y = df["label"].values
    label_names = sorted(np.unique(y))

    results = {}

    # RandomForest
    acc_rf, cm_rf, rep_rf, trues, preds_rf, _ = _loocv(
        RandomForestClassifier,
        {"n_estimators": RF_N_ESTIMATORS, "class_weight": "balanced", "n_jobs": -1},
        X, y, label_names,
    )
    results["RandomForest"] = {"acc": acc_rf, "cm": cm_rf, "report": rep_rf}
    print(f"\n[RandomForest] LOOCV Accuracy = {acc_rf:.3f}")
    print(rep_rf)

    # SVM
    best_acc_svm = 0
    best_cfg = {}
    for C in SVM_C_GRID:
        for gamma in SVM_GAMMA_GRID:
            acc_svm, cm_svm, rep_svm, _, _, _ = _loocv(
                SVC,
                {"C": C, "gamma": gamma, "kernel": "rbf", "class_weight": "balanced", "probability": True},
                X, y, label_names,
            )
            if acc_svm > best_acc_svm:
                best_acc_svm = acc_svm
                best_cfg = {"C": C, "gamma": gamma}
    results["SVM"] = {"acc": best_acc_svm, "best_cfg": best_cfg}
    print(f"\n[SVM] Best LOOCV Accuracy = {best_acc_svm:.3f} (C={best_cfg['C']}, gamma={best_cfg['gamma']})")

    # kNN
    acc_knn, cm_knn, rep_knn, _, _, _ = _loocv(
        KNeighborsClassifier,
        {"n_neighbors": KNN_K},
        X, y, label_names,
    )
    results["kNN"] = {"acc": acc_knn, "cm": cm_knn, "report": rep_knn}
    print(f"\n[kNN] LOOCV Accuracy = {acc_knn:.3f}")

    # 保存混淆矩阵图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        if "cm" not in res:
            continue
        cm = res["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}\nAcc={res['acc']:.3f}")
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # 保存错误分析
    errors = []
    for i, (t, p) in enumerate(zip(trues, preds_rf)):
        if t != p:
            idx = df.iloc[i]["idx"]
            errors.append({"idx": int(idx), "true": t, "pred": p, "model": "RandomForest"})
    pd.DataFrame(errors).to_csv(out_dir / "error_analysis.csv", index=False)

    # 保存最终模型（全量训练）
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    final_clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    final_clf.fit(X_s, y)
    pipeline = {
        "model": final_clf,
        "scaler": scaler,
        "feature_cols": selected_cols,
        "label_names": label_names,
    }
    joblib.dump(pipeline, out_dir / "pipeline.joblib")
    print(f"\n[Model] Saved -> {out_dir / 'pipeline.joblib'}")

    # 保存 JSON 报告
    best_acc = max(res["acc"] for res in results.values() if "acc" in res)
    summary = {
        "n_samples": len(df),
        "n_features": len(selected_cols),
        "label_distribution": {name: int((y == name).sum()) for name in label_names},
        "results": {name: {"accuracy": float(res["acc"])} for name, res in results.items() if "acc" in res},
        "best_model": max(results, key=lambda k: results[k].get("acc", 0)),
        "best_accuracy": float(best_acc),
        "target_accuracy": TARGET_LOOCV_ACC,
        "pass": best_acc >= TARGET_LOOCV_ACC,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary
