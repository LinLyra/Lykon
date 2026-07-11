"""
features.py — 特征提取

对每个片段的每个通道提取时域 + 频域特征，并计算跨节点特征。
输出: DataFrame (N_segments × N_features) 保存为 parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

from config import FS, FFT_N_PER_SEG, FFT_BANDS, FEATURE_LIST, FEATURE_FIND_PEAKS_DISTANCE


def _dominant_freq(x: np.ndarray, fs: int = FS) -> tuple[float, float, float]:
    """返回 (主频 Hz, 主频幅值, 总能量)"""
    n = len(x)
    w = np.hanning(n)
    spec = np.abs(rfft(x * w))
    freqs = rfftfreq(n, d=1.0 / fs)
    total = np.sum(spec ** 2)
    if len(spec) <= 1 or total == 0:
        return 0.0, 0.0, 1e-12
    idx = np.argmax(spec[1:]) + 1
    return float(freqs[idx]), float(spec[idx]), float(total)


def _spectral_features(x: np.ndarray, fs: int = FS) -> dict:
    dom_freq, dom_amp, total = _dominant_freq(x, fs)
    n = len(x)
    w = np.hanning(n)
    spec = np.abs(rfft(x * w)) ** 2
    freqs = rfftfreq(n, d=1.0 / fs)

    # 频谱质心
    spec_safe = spec + 1e-12
    centroid = np.sum(freqs * spec_safe) / np.sum(spec_safe)

    # 频谱熵
    p = spec_safe / np.sum(spec_safe)
    entropy = -np.sum(p * np.log2(p + 1e-12))

    # 频段能量比
    bands = {}
    for (low, high) in FFT_BANDS:
        mask = (freqs >= low) & (freqs < high)
        bands[f"band_{low}_{high}"] = float(np.sum(spec[mask]) / total)

    return {
        "dom_freq": dom_freq,
        "dom_freq_ratio": float(dom_amp ** 2 / total),
        "spectral_centroid": float(centroid),
        "spectral_entropy": float(entropy),
        **bands,
    }


def _time_features(x: np.ndarray, fs: int = FS) -> dict:
    peaks, _ = find_peaks(np.abs(x), distance=max(1, FEATURE_FIND_PEAKS_DISTANCE))
    zero_cross = np.sum(np.diff(np.sign(x)) != 0)
    sma = np.sum(np.abs(x)) / len(x)
    jerk = np.diff(x) * fs if len(x) > 1 else np.array([0.0])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
        "ptp": float(np.ptp(x)),
        "zero_crossing_rate": int(zero_cross),
        "n_peaks": int(len(peaks)),
        "skewness": float(stats.skew(x)) if len(x) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(x)) if len(x) > 2 else 0.0,
        "sma": float(sma),
        "jerk_max": float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0,
    }


# ──────────────────────────────
# 主入口：从 NPZ 提取特征
# ──────────────────────────────
def extract_features(npz_path: str) -> pd.DataFrame:
    """
    输入 segments.npz: X[N, T, C], y[N], meta[N], feature_cols[C]
    输出 DataFrame: 每行一个片段，列为特征
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    meta = data["meta"]
    feature_cols = list(data["feature_cols"])

    N, T, C = X.shape
    print(f"[Features] 输入: {N} segments, {T} timepoints, {C} channels")

    rows = []
    for i in range(N):
        seg = X[i]
        feats = {}
        # 逐通道特征
        for c, col in enumerate(feature_cols):
            ch = seg[:, c]
            t_feats = _time_features(ch, FS)
            f_feats = _spectral_features(ch, FS)
            for k, v in {**t_feats, **f_feats}.items():
                feats[f"{col}_{k}"] = v

        # 跨节点特征（硬编码关键通道）
        # 需要找到对应的列索引
        def _idx(name):
            return feature_cols.index(name) if name in feature_cols else None

        # RF a_norm vs RU a_norm 相关系数
        rf_a = _idx("RF_a_norm")
        ru_a = _idx("RU_a_norm")
        if rf_a is not None and ru_a is not None:
            feats["corr_RF_RU_a_norm"] = float(np.corrcoef(seg[:, rf_a], seg[:, ru_a])[0, 1])

        # RF a_norm_hp vs RU a_norm_hp 相关系数
        rf_a_hp = _idx("RF_a_norm_hp")
        ru_a_hp = _idx("RU_a_norm_hp")
        if rf_a_hp is not None and ru_a_hp is not None:
            feats["corr_RF_RU_a_norm_hp"] = float(np.corrcoef(seg[:, rf_a_hp], seg[:, ru_a_hp])[0, 1])

        # RF a_norm vs LF a_norm 相关系数
        lf_a = _idx("LF_a_norm")
        if rf_a is not None and lf_a is not None:
            feats["corr_RF_LF_a_norm"] = float(np.corrcoef(seg[:, rf_a], seg[:, lf_a])[0, 1])

        # RF a_norm vs RU jerk 相关系数
        ru_jerk = _idx("RU_jerk")
        if rf_a is not None and ru_jerk is not None:
            feats["corr_RF_RU_jerk"] = float(np.corrcoef(seg[:, rf_a], seg[:, ru_jerk])[0, 1])

        # LF-RF 能量比 (RMS)
        if lf_a is not None and rf_a is not None:
            feats["energy_ratio_LF_RF"] = float(np.sqrt(np.mean(seg[:, lf_a] ** 2)) / (np.sqrt(np.mean(seg[:, rf_a] ** 2)) + 1e-12))

        # RF 高通峰值计数（拍球的冲击峰）
        rf_a_hp = _idx("RF_a_norm_hp")
        if rf_a_hp is not None:
            rf_hp_sig = seg[:, rf_a_hp]
            peaks, _ = find_peaks(np.abs(rf_hp_sig), distance=max(1, int(FS * 0.05)), prominence=np.std(rf_hp_sig))
            feats["RF_hp_peak_count"] = int(len(peaks))

        # RF 高频带能量占比
        if rf_a_hp is not None:
            f_feats = _spectral_features(seg[:, rf_a_hp], FS)
            feats["RF_hp_band_8_25"] = f_feats.get("band_8_25", 0.0)

        feats["label"] = str(y[i])
        feats["idx"] = i
        feats["cycle"] = int(meta[i]["cycle"]) if isinstance(meta[i], dict) else int(meta[i]["cycle"])
        feats["suspicious"] = bool(meta[i]["suspicious"]) if isinstance(meta[i], dict) else bool(meta[i]["suspicious"])
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f"[Features] 输出: {df.shape} (包含 label 列)")
    return df


def select_features(df: pd.DataFrame, n_top: int = 40, random_state: int = 42) -> list[str]:
    """
    使用随机森林特征重要性 + t-test 排序，选出 top N 特征。
    返回特征名列表（不含 label/idx/cycle/suspicious）。
    """
    from sklearn.ensemble import RandomForestClassifier

    drop_cols = ["label", "idx", "cycle", "suspicious"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # 只保留数值列
    X = X.select_dtypes(include=[np.number])
    y = df["label"].values

    # 移除常数列
    X = X.loc[:, X.std() > 0]

    # RF 重要性
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # t-test 排序（每特征对两类的区分力）
    t_scores = {}
    for col in X.columns:
        a = X.loc[y == "dribble", col].values
        b = X.loc[y == "shot", col].values
        if len(a) > 1 and len(b) > 1:
            t_scores[col] = abs(stats.ttest_ind(a, b).statistic)
    t_scores = pd.Series(t_scores).sort_values(ascending=False)

    # 综合排序：取两者排名的平均
    combined = {}
    for col in X.columns:
        r_rf = list(importances.index).index(col) if col in importances.index else len(X.columns)
        r_t = list(t_scores.index).index(col) if col in t_scores.index else len(X.columns)
        combined[col] = (r_rf + r_t) / 2
    combined = pd.Series(combined).sort_values()

    selected = list(combined.head(n_top).index)
    print(f"[Select] 从 {len(X.columns)} 维选出 {n_top} 维: {', '.join(selected[:5])}, ...")
    return selected
