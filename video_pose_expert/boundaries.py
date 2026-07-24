"""各动作的识别边界（decision boundary per action）。

对每个动作，在标准化特征空间里：
    PCA 降维 → 拟合协方差（LedoitWolf，鲁棒于小样本）→ Mahalanobis 距离
    → 阈值 = sqrt(chi2.ppf(q, df=k))

于是每个动作得到一个"识别边界椭球"：
    - 测试样本到该动作分布的 Mahalanobis 距离 ≤ 阈值 ⇒ 落在该动作专家模式内
    - 距离越小 ⇒ 与专家动作模式越接近（可换算成匹配度分数）

这满足需求中"训练出各个动作的识别边界，以作为 expert 对任一人员的
测试数据进行识别"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from .config import BOUNDARY_CHI2_Q, BOUNDARY_PCA_DIM


@dataclass
class ActionBoundary:
    """单个动作在标准化特征空间中的识别边界。"""
    action: str
    pca: PCA
    mean: np.ndarray          # PCA 空间均值 (k,)
    precision: np.ndarray     # 协方差的逆 (k, k)
    threshold: float          # Mahalanobis 距离阈值
    k: int
    n_samples: int

    def distance(self, Xs: np.ndarray) -> np.ndarray:
        """标准化特征 Xs (n, d) 到本动作分布的 Mahalanobis 距离 (n,)。"""
        z = self.pca.transform(Xs) - self.mean[None, :]
        d2 = np.einsum("ij,jk,ik->i", z, self.precision, z)
        return np.sqrt(np.clip(d2, 0.0, None))

    def within(self, Xs: np.ndarray) -> np.ndarray:
        return self.distance(Xs) <= self.threshold

    def match_score(self, Xs: np.ndarray) -> np.ndarray:
        """0~100 匹配度：距离=0 → 100；距离=阈值 → ~50；越远越低。"""
        d = self.distance(Xs)
        return 100.0 / (1.0 + (d / max(self.threshold, 1e-6)) ** 2)


def fit_action_boundary(action: str, Xs_action: np.ndarray,
                        pca_dim: int = BOUNDARY_PCA_DIM,
                        q: float = BOUNDARY_CHI2_Q) -> ActionBoundary:
    """在某动作的标准化特征样本上拟合识别边界。"""
    n, d = Xs_action.shape
    k = int(min(pca_dim, d, max(1, n - 1)))
    pca = PCA(n_components=k, random_state=0).fit(Xs_action)
    z = pca.transform(Xs_action)
    mean = z.mean(axis=0)
    cov = LedoitWolf().fit(z).covariance_
    precision = np.linalg.pinv(cov)
    threshold = float(np.sqrt(chi2.ppf(q, df=k)))
    return ActionBoundary(action=action, pca=pca, mean=mean,
                          precision=precision, threshold=threshold,
                          k=k, n_samples=n)


def fit_all_boundaries(Xs: np.ndarray, y: np.ndarray,
                       pca_dim: int = BOUNDARY_PCA_DIM,
                       q: float = BOUNDARY_CHI2_Q,
                       min_samples: int = 4) -> dict[str, ActionBoundary]:
    """对每个动作拟合边界。样本过少的动作会被跳过（并告警）。"""
    boundaries: dict[str, ActionBoundary] = {}
    for action in np.unique(y):
        mask = y == action
        n = int(mask.sum())
        if n < min_samples:
            print(f"[boundary] 跳过 '{action}'：样本仅 {n} 个（< {min_samples}）")
            continue
        boundaries[str(action)] = fit_action_boundary(str(action), Xs[mask], pca_dim, q)
    return boundaries
