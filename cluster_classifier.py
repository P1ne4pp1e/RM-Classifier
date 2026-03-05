"""
cluster_classifier.py - 基于 K-Means 过聚类的开放集图像分类器

核心设计思路：
  - 只对正常类建模（每类独立 K-Means，k >> 1 以捕获类内多样性）
  - 通过到最近质心的距离阈值实现开放集拒绝
  - 马氏距离作为辅助拒绝信号（考虑类内分布形状）
  - None 类不参与聚类建模，仅用于训练时报告参考

推理复杂度：一次矩阵乘法 + argmin，单张 < 0.01ms（Intel CPU）
"""

import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import joblib

from features import FeatureExtractor


class ClusterClassifier:
    """
    开放集聚类分类器。

    接口：
      train(normal_data, none_data)  → 训练并校准阈值
      predict_single(image)          → (class_name, distance, is_rejected)
      predict_batch(images)          → list of (class_name, distance, is_rejected)
      save(path) / load(path)        → 模型持久化
    """

    def __init__(self, config: dict, extractor: FeatureExtractor):
        self.config = config
        self.extractor = extractor

        # 正常类信息
        self.class_names: list[str] = []           # 正常类名称列表
        self.none_class: str = config['dataset']['none_class_name']

        # K-Means 质心（训练后填充）
        self.all_centroids: np.ndarray | None = None   # (K_total, D) float32
        self.centroid_labels: np.ndarray | None = None  # (K_total,) int32，每个质心对应的类别索引

        # 马氏距离参数（每个正常类一份）
        self.class_means: dict[str, np.ndarray] = {}    # class → (D,)
        self.class_cov_inv: dict[str, np.ndarray] = {}  # class → (D, D)

        # 阈值（校准后填充）
        self.distance_threshold: float | None = None    # 全局欧氏距离阈值
        self.mahal_thresholds: dict[str, float] = {}    # 每类马氏距离阈值

    # ------------------------------------------------------------------ #
    #  训练                                                                 #
    # ------------------------------------------------------------------ #

    def train(
        self,
        normal_data: dict[str, list[np.ndarray]],
        none_data: list[np.ndarray] | None = None,
        val_data: dict[str, list[np.ndarray]] | None = None,
    ) -> None:
        """
        Args:
            normal_data: {class_name: [img, ...]}，正常类训练图像
            none_data:   [img, ...]，None 类图像（仅用于报告，不参与建模）
            val_data:    {class_name: [img, ...]}，正常类验证图像
                         若提供则用验证集校准阈值（推荐），否则回退用训练集
        """
        cfg_c = self.config['clustering']
        cfg_r = self.config['rejection']
        verbose = self.config['output']['verbose']

        self.class_names = sorted(normal_data.keys())

        # ── 1. 提取正常类特征 ─────────────────────────────────────────────
        if verbose:
            print("[训练] 提取特征...")

        class_feats: dict[str, np.ndarray] = {}
        for cls in self.class_names:
            feats = self.extractor.transform(normal_data[cls])
            class_feats[cls] = feats
            if verbose:
                print(f"  {cls}: {len(normal_data[cls])} 样本 → {feats.shape}")

        # ── 2. 每类独立 K-Means（过聚类）────────────────────────────────
        if verbose:
            print(f"[训练] K-Means 聚类 (k={cfg_c['k_per_class']} / 类)...")

        k = cfg_c['k_per_class']
        all_centroids = []
        all_labels = []

        for cls_idx, cls in enumerate(self.class_names):
            feats = class_feats[cls]
            # 安全处理：样本数不足时减少 k
            actual_k = min(k, len(feats))

            km = KMeans(
                n_clusters=actual_k,
                n_init=cfg_c['n_init'],
                max_iter=cfg_c['max_iter'],
                random_state=cfg_c['random_seed'],
            )
            km.fit(feats)
            all_centroids.append(km.cluster_centers_)
            all_labels.extend([cls_idx] * actual_k)

            if verbose:
                avg_inertia = km.inertia_ / len(feats)
                print(f"  {cls}: k={actual_k}, 平均惯性={avg_inertia:.4f}")

        self.all_centroids = np.vstack(all_centroids).astype(np.float32)
        self.centroid_labels = np.array(all_labels, dtype=np.int32)

        # ── 3. 马氏距离参数（每类协方差矩阵）────────────────────────────
        if cfg_r['mahalanobis']['enabled']:
            reg = cfg_r['mahalanobis']['regularization']
            D = self.all_centroids.shape[1]
            for cls in self.class_names:
                feats = class_feats[cls]
                mu = feats.mean(axis=0)
                cov = np.cov(feats, rowvar=False)
                cov += np.eye(D) * reg  # 正则化防止奇异
                self.class_means[cls] = mu.astype(np.float32)
                self.class_cov_inv[cls] = np.linalg.pinv(cov).astype(np.float32)

        # ── 4. 阈值校准 ────────────────────────────────────────────────────
        # 优先用验证集（unseen data）校准，避免训练集"自己人近"带来的偏差
        if verbose:
            if val_data:
                print("[训练] 校准拒绝阈值（使用验证集）...")
            else:
                print("[训练] 校准拒绝阈值（使用训练集，建议提供 val_data）...")

        if val_data:
            calib_feats = np.vstack([
                self.extractor.transform(val_data[cls])
                for cls in self.class_names
                if cls in val_data
            ])
        else:
            calib_feats = np.vstack(list(class_feats.values()))

        min_dists = self._batch_min_distances(calib_feats)
        p = cfg_r['distance_percentile']
        self.distance_threshold = float(np.percentile(min_dists, p))

        if cfg_r['mahalanobis']['enabled']:
            p_m = cfg_r['mahalanobis']['percentile']
            for cls in self.class_names:
                if val_data and cls in val_data:
                    calib_cls_feats = self.extractor.transform(val_data[cls])
                else:
                    calib_cls_feats = class_feats[cls]
                mdists = self._batch_mahal_distances(calib_cls_feats, cls)
                self.mahal_thresholds[cls] = float(np.percentile(mdists, p_m))

        if verbose:
            print(f"  欧氏距离阈值 ({p}分位数): {self.distance_threshold:.4f}")
            if cfg_r['mahalanobis']['enabled']:
                for cls, t in self.mahal_thresholds.items():
                    p_m = cfg_r['mahalanobis']['percentile']
                    print(f"  马氏距离阈值 {cls} ({p_m}分位数): {t:.4f}")

        # ── 5. 参考报告：None 类拒绝率 ────────────────────────────────────
        if none_data and verbose:
            print("[训练] 评估 None 类检测率（参考，不影响阈值）...")
            none_feats = self.extractor.transform(none_data)
            rejected = self._batch_rejected(none_feats)
            rate = rejected.mean() * 100
            print(f"  None 拒绝率（训练已见）: {rate:.1f}%  "
                  f"({int(rejected.sum())}/{len(none_data)})")

    # ------------------------------------------------------------------ #
    #  推理                                                                 #
    # ------------------------------------------------------------------ #

    def predict_single(
        self, image: np.ndarray
    ) -> tuple[str, float, bool]:
        """
        对单张图像推理。

        Returns:
            class_name:  预测类别（若被拒绝则为 none_class_name）
            distance:    到最近质心的欧氏距离（可作为置信度参考，越小越确定）
            is_rejected: True 表示判定为 None
        """
        feat = self.extractor.transform_single(image)[0]  # (D,)
        return self._predict_feat(feat)

    def predict_batch(
        self, images: list[np.ndarray]
    ) -> list[tuple[str, float, bool]]:
        """批量推理，返回 [(class_name, distance, is_rejected), ...]"""
        feats = self.extractor.transform(images)  # (N, D)
        return [self._predict_feat(feats[i]) for i in range(len(feats))]

    # ------------------------------------------------------------------ #
    #  内部推理逻辑                                                          #
    # ------------------------------------------------------------------ #

    def _predict_feat(self, feat: np.ndarray) -> tuple[str, float, bool]:
        """feat: (D,) 已归一化特征向量"""
        # 计算到所有质心的平方距离（利用向量化加速）
        diff = self.all_centroids - feat  # (K, D)
        sq_dists = np.einsum('kd,kd->k', diff, diff)  # (K,)
        nearest_idx = int(np.argmin(sq_dists))
        min_dist = float(np.sqrt(max(sq_dists[nearest_idx], 0.0)))

        cls_idx = int(self.centroid_labels[nearest_idx])
        class_name = self.class_names[cls_idx]

        # 欧氏距离拒绝
        if min_dist > self.distance_threshold:
            return self.none_class, min_dist, True

        # 马氏距离拒绝（辅助）
        if (self.config['rejection']['mahalanobis']['enabled']
                and class_name in self.mahal_thresholds):
            mdist = self._single_mahal_distance(feat, class_name)
            if mdist > self.mahal_thresholds[class_name]:
                return self.none_class, min_dist, True

        return class_name, min_dist, False

    # ------------------------------------------------------------------ #
    #  距离计算工具                                                          #
    # ------------------------------------------------------------------ #

    def _batch_min_distances(self, feats: np.ndarray) -> np.ndarray:
        """
        计算每个样本到最近质心的欧氏距离。
        利用 ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c^T 的矩阵乘法展开。
        """
        x_sq = np.sum(feats ** 2, axis=1, keepdims=True)     # (N, 1)
        c_sq = np.sum(self.all_centroids ** 2, axis=1)         # (K,)
        cross = feats @ self.all_centroids.T                   # (N, K)
        sq_dists = x_sq + c_sq - 2 * cross                    # (N, K)
        sq_dists = np.maximum(sq_dists, 0.0)                   # 数值保护
        return np.sqrt(np.min(sq_dists, axis=1))               # (N,)

    def _batch_mahal_distances(self, feats: np.ndarray, cls: str) -> np.ndarray:
        """计算样本到指定类的马氏距离（批量）"""
        mu = self.class_means[cls]
        cov_inv = self.class_cov_inv[cls]
        diff = feats - mu                       # (N, D)
        temp = diff @ cov_inv                   # (N, D)
        mdists_sq = np.einsum('nd,nd->n', temp, diff)  # (N,)
        return np.sqrt(np.maximum(mdists_sq, 0.0))

    def _single_mahal_distance(self, feat: np.ndarray, cls: str) -> float:
        """计算单个样本到指定类的马氏距离"""
        mu = self.class_means[cls]
        cov_inv = self.class_cov_inv[cls]
        diff = feat - mu
        return float(np.sqrt(max(diff @ cov_inv @ diff, 0.0)))

    def _batch_rejected(self, feats: np.ndarray) -> np.ndarray:
        """批量判断是否被拒绝（bool 数组）"""
        min_dists = self._batch_min_distances(feats)
        rejected = min_dists > self.distance_threshold

        if self.config['rejection']['mahalanobis']['enabled']:
            # 对尚未被欧氏距离拒绝的样本，补充马氏距离检查
            x_sq = np.sum(feats ** 2, axis=1, keepdims=True)
            c_sq = np.sum(self.all_centroids ** 2, axis=1)
            cross = feats @ self.all_centroids.T
            sq_dists = np.maximum(x_sq + c_sq - 2 * cross, 0.0)
            nearest_idx = np.argmin(sq_dists, axis=1)
            cls_idxs = self.centroid_labels[nearest_idx]

            not_rejected = ~rejected
            for i in np.where(not_rejected)[0]:
                cls = self.class_names[int(cls_idxs[i])]
                if cls in self.mahal_thresholds:
                    mdist = self._single_mahal_distance(feats[i], cls)
                    if mdist > self.mahal_thresholds[cls]:
                        rejected[i] = True

        return rejected

    # ------------------------------------------------------------------ #
    #  模型持久化                                                            #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> 'ClusterClassifier':
        classifier = joblib.load(path)
        print(f"模型已加载: {path}")
        return classifier
