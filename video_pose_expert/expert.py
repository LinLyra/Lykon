"""Expert：训练 + 推理。

训练：姿态特征 → 标准化 → (a) 多类识别器 RandomForest
                              (b) 各动作识别边界 ActionBoundary
推理：对任意人员的测试数据给出
        predicted_action  预测动作
        class_proba        分类器置信度
        boundary_distance  到所预测动作专家分布的 Mahalanobis 距离
        within_boundary    是否落在该动作识别边界内
        expert_match_score 与专家动作模式的匹配度 (0~100)
      以及到每个动作的匹配度分布（match_<action>）。
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE, TEST_SIZE, DEFAULT_FPS
from .boundaries import fit_all_boundaries, ActionBoundary
from .features import features_dataframe, feature_columns


class Expert:
    def __init__(self, scaler: StandardScaler, classifier: RandomForestClassifier,
                 feature_cols: list[str], boundaries: dict[str, ActionBoundary],
                 fps: float = DEFAULT_FPS):
        self.scaler = scaler
        self.classifier = classifier
        self.feature_cols = feature_cols
        self.boundaries = boundaries
        self.fps = fps

    # ---------------- 持久化 ----------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "classifier": self.classifier,
            "feature_cols": self.feature_cols,
            "boundaries": self.boundaries,
            "fps": self.fps,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "Expert":
        d = joblib.load(path)
        return cls(d["scaler"], d["classifier"], d["feature_cols"],
                   d["boundaries"], d.get("fps", DEFAULT_FPS))

    # ---------------- 推理 ----------------
    def _matrix(self, feat_df: pd.DataFrame) -> np.ndarray:
        X = feat_df.reindex(columns=self.feature_cols).fillna(0.0).to_numpy(dtype=float)
        return self.scaler.transform(X)

    def predict_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """输入特征 DataFrame，返回带预测结果的 DataFrame。"""
        Xs = self._matrix(feat_df)
        proba = self.classifier.predict_proba(Xs)
        classes = list(self.classifier.classes_)
        pred_idx = proba.argmax(axis=1)
        pred = np.array([classes[i] for i in pred_idx])

        meta_cols = [c for c in feat_df.columns if c not in self.feature_cols]
        out = feat_df[meta_cols].copy().reset_index(drop=True)
        out["predicted_action"] = pred
        out["class_proba"] = proba.max(axis=1)

        # 到每个动作边界的匹配度
        match = {a: b.match_score(Xs) for a, b in self.boundaries.items()}
        dist = {a: b.distance(Xs) for a, b in self.boundaries.items()}
        within = {a: b.within(Xs) for a, b in self.boundaries.items()}
        for a in self.boundaries:
            out[f"match_{a}"] = match[a]

        # 针对"所预测动作"的边界指标
        bd = np.full(len(pred), np.nan)
        wb = np.zeros(len(pred), dtype=bool)
        ms = np.zeros(len(pred))
        for i, a in enumerate(pred):
            if a in self.boundaries:
                bd[i] = dist[a][i]
                wb[i] = within[a][i]
                ms[i] = match[a][i]
        out["boundary_distance"] = bd
        out["within_boundary"] = wb
        out["expert_match_score"] = ms
        return out

    def predict_samples(self, samples: list[dict], fps: float | None = None) -> pd.DataFrame:
        """输入姿态样本列表（含 seq），自动提特征再推理。"""
        feat_df = features_dataframe(samples, fps=fps or self.fps)
        return self.predict_features(feat_df)

    # ---------------- 汇总 ----------------
    def summarize(self, pred_df: pd.DataFrame) -> dict:
        """把逐样本预测汇总成人员级报告。"""
        counts = pred_df["predicted_action"].value_counts().to_dict()
        return {
            "n_samples": int(len(pred_df)),
            "action_distribution": {k: int(v) for k, v in counts.items()},
            "mean_class_proba": float(pred_df["class_proba"].mean()),
            "mean_expert_match": float(pred_df["expert_match_score"].mean()),
            "within_boundary_rate": float(pred_df["within_boundary"].mean()),
        }


# ------------------------------------------------------------------ 训练
def train_expert(features_df: pd.DataFrame, out_dir: str | Path,
                 fps: float = DEFAULT_FPS) -> dict:
    """训练 Expert 并落盘。返回评估指标 dict。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(features_df)
    X = features_df[cols].fillna(0.0).to_numpy(dtype=float)
    y = features_df["label"].astype(str).to_numpy()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # (a) 多类识别器
    n_classes = len(np.unique(y))
    stratify = y if (n_classes > 1 and np.min(np.bincount(
        pd.factorize(y)[0])) >= 2) else None
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)
    clf = RandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    labels_sorted = sorted(np.unique(y).tolist())
    report_txt = classification_report(yte, pred, labels=labels_sorted, zero_division=0)
    report_dict = classification_report(
        yte, pred, labels=labels_sorted, zero_division=0, output_dict=True)
    cm = confusion_matrix(yte, pred, labels=labels_sorted)

    # (b) 各动作识别边界（用全部数据拟合，边界描述"专家动作模式")
    boundaries = fit_all_boundaries(Xs, y)

    # 在整个训练集上用完整数据重训分类器（边界+分类器都作为专家知识）
    clf_full = RandomForestClassifier(
        n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    clf_full.fit(Xs, y)

    expert = Expert(scaler, clf_full, cols, boundaries, fps=fps)
    expert.save(out_dir / "expert.joblib")

    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        out_dir / "confusion_matrix.csv")
    boundary_summary = pd.DataFrame([
        {"action": a, "n_samples": b.n_samples, "pca_dim": b.k,
         "threshold": round(b.threshold, 3)}
        for a, b in boundaries.items()
    ])
    boundary_summary.to_csv(out_dir / "boundary_summary.csv", index=False)

    return {
        "n_samples": int(len(features_df)),
        "n_features": len(cols),
        "classes": labels_sorted,
        "test_accuracy": float(report_dict["accuracy"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "report_text": report_txt,
        "boundaries": {a: {"threshold": round(b.threshold, 3), "n": b.n_samples}
                       for a, b in boundaries.items()},
        "expert_path": str(out_dir / "expert.joblib"),
    }
