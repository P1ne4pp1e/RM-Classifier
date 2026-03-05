"""
features.py - 特征提取器

从 32x32 二值图像（uint8, 像素值 0/255）中提取多维特征向量。

特征组成（按重要性排序）：
  1. PCA 降维原始像素  — 保留所有空间细节，是主力特征
  2. Hu 矩            — 全局形状不变量（7维）
  3. 轮廓傅里叶描述子  — 形状边界的频域细节（n维）
  4. 拓扑特征          — 连通域数/Euler数/面积比/宽高比（4维）
  5. LBP 直方图        — 局部纹理（可选，59维）
  6. 多尺度密度网格    — 空间像素分布（可选）

使用方式：
  extractor = FeatureExtractor(config['features'])
  extractor.fit(normal_images)           # 仅用正常类数据拟合 PCA 和 Scaler
  feats = extractor.transform(images)    # 返回 (N, D) float32
"""

import warnings
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    多维特征提取器，支持 PCA 降维和特征归一化。

    fit()     → 仅用正常类图像拟合，确保 PCA 和 Scaler 不受 None 类污染
    transform() → 对任意图像批量提取+归一化特征
    """

    def __init__(self, config: dict):
        self.config = config
        self.pca: PCA | None = None
        self.scaler: StandardScaler | None = None
        self._fitted = False
        self._other_dim: int | None = None  # 缓存非PCA特征维度

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, images: list[np.ndarray], verbose: bool = False) -> 'FeatureExtractor':
        """
        用正常类图像拟合 PCA 和 StandardScaler。

        Args:
            images: 正常类图像列表，每张为 (32, 32) uint8 ndarray
            verbose: 是否打印特征维度信息
        """
        raw_pixels, other_feats = self._batch_extract(images)

        if self.config['pca']['enabled']:
            ratio = self.config['pca']['variance_ratio']
            self.pca = PCA(n_components=ratio, random_state=42)
            pca_feats = self.pca.fit_transform(raw_pixels)
            all_feats = np.concatenate([pca_feats, other_feats], axis=1)
        else:
            all_feats = other_feats

        self.scaler = StandardScaler()
        self.scaler.fit(all_feats)
        self._fitted = True

        if verbose:
            n_pca = int(self.pca.n_components_) if self.pca else 0
            n_other = other_feats.shape[1]
            total = n_pca + n_other
            print(f"  特征维度: PCA={n_pca} + 其他={n_other} = {total} 维")

        return self

    def transform(self, images: list[np.ndarray]) -> np.ndarray:
        """
        提取并归一化特征。

        Args:
            images: 图像列表，每张为 (32, 32) uint8 ndarray

        Returns:
            feats: (N, D) float32 ndarray，已归一化
        """
        assert self._fitted, "请先调用 fit()"
        raw_pixels, other_feats = self._batch_extract(images)

        if self.config['pca']['enabled']:
            pca_feats = self.pca.transform(raw_pixels)
            all_feats = np.concatenate([pca_feats, other_feats], axis=1)
        else:
            all_feats = other_feats

        return self.scaler.transform(all_feats).astype(np.float32)

    def transform_single(self, image: np.ndarray) -> np.ndarray:
        """提取单张图像特征，返回 (1, D) float32"""
        return self.transform([image])

    def feature_dim(self) -> int:
        """返回最终特征向量维度"""
        assert self._fitted
        n_pca = int(self.pca.n_components_) if self.pca else 0
        n_other = self._get_other_dim()
        return n_pca + n_other

    # ------------------------------------------------------------------ #
    #  批量提取（内部）                                                      #
    # ------------------------------------------------------------------ #

    def _batch_extract(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """提取所有图像的原始像素和其他特征，分开返回便于 PCA 处理"""
        raw_list = []
        other_list = []
        for img in images:
            raw_list.append(_extract_raw_pixels(img))
            other_list.append(self._extract_other(img))
        return np.array(raw_list, dtype=np.float32), np.array(other_list, dtype=np.float32)

    def _extract_other(self, img: np.ndarray) -> np.ndarray:
        """提取所有非原始像素的特征并拼接"""
        parts = []

        if self.config['hu_moments']['enabled']:
            parts.append(_extract_hu_moments(img))

        if self.config['fourier_descriptors']['enabled']:
            n = self.config['fourier_descriptors']['n_descriptors']
            parts.append(_extract_fourier_descriptors(img, n))

        if self.config['topology']['enabled']:
            parts.append(_extract_topology(img))

        if self.config['lbp']['enabled']:
            r = self.config['lbp']['radius']
            p = self.config['lbp']['n_points']
            parts.append(_extract_lbp(img, r, p))

        if self.config['density_grid']['enabled']:
            scales = self.config['density_grid']['scales']
            parts.append(_extract_density_grid(img, scales))

        if not parts:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _get_other_dim(self) -> int:
        if self._other_dim is None:
            dummy = np.zeros((32, 32), dtype=np.uint8)
            self._other_dim = len(self._extract_other(dummy))
        return self._other_dim


# ====================================================================== #
#  静态特征函数（无状态，便于单独调用/测试）                                  #
# ====================================================================== #

def _extract_raw_pixels(img: np.ndarray) -> np.ndarray:
    """展平并归一化到 [0, 1]，返回 (1024,) float32"""
    return img.flatten().astype(np.float32) / 255.0


def _extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """
    计算 7 个 Hu 矩，并做 log 变换压缩数值范围。
    对平移、缩放、旋转不变。
    """
    binary = (img > 127).astype(np.uint8) * 255
    m = cv2.moments(binary)
    hu = cv2.HuMoments(m).flatten()
    # log 变换：压缩量级差异，保留符号
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)


def _extract_fourier_descriptors(img: np.ndarray, n: int = 20) -> np.ndarray:
    """
    轮廓傅里叶描述子（n维）。

    原理：
      - 将最大轮廓表示为复数序列 z = x + jy
      - 对 z 做 FFT，取前 n 个系数的模
      - 除以 |Z[1]| 实现缩放不变性（Z[0] 为均值，Z[1] 最大）
      - 取模后旋转不变，去掉 Z[0] 后平移不变

    退化处理（无轮廓 / 轮廓点不足）：返回零向量。
    """
    binary = (img > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros(n, dtype=np.float32)

    # 取最大轮廓
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

    if len(contour) < n + 1:
        return np.zeros(n, dtype=np.float32)

    z = contour[:, 0] + 1j * contour[:, 1]
    Z = np.fft.fft(z)

    # Z[1] 用于归一化（若接近 0 说明轮廓退化）
    if abs(Z[1]) < 1e-10:
        return np.zeros(n, dtype=np.float32)

    # 跳过 Z[0]（DC/平移分量），取 Z[1..n] 的归一化模
    descriptors = np.abs(Z[1:n + 1]) / np.abs(Z[1])

    if len(descriptors) < n:
        descriptors = np.pad(descriptors, (0, n - len(descriptors)))

    return descriptors.astype(np.float32)


def _extract_topology(img: np.ndarray) -> np.ndarray:
    """
    拓扑特征（4维）：
      [0] 白色像素占比（面积比）
      [1] 连通域数量
      [2] Euler 数（连通域数 - 孔洞数）
      [3] 轮廓边界框宽高比
    """
    from skimage.measure import label as sk_label, euler_number

    binary = img > 127

    # 面积比
    area_ratio = float(np.mean(binary))

    # 连通域数量
    labeled = sk_label(binary, connectivity=2)
    n_components = float(labeled.max())

    # Euler 数（拓扑不变量）
    try:
        euler = float(euler_number(binary, connectivity=2))
    except Exception:
        euler = 0.0

    # 轮廓边界框宽高比
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        h = float(rmax - rmin + 1)
        w = float(cmax - cmin + 1)
        aspect = w / (h + 1e-5)
    else:
        aspect = 1.0

    return np.array([area_ratio, n_components, euler, aspect], dtype=np.float32)


def _extract_lbp(img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    LBP uniform 模式直方图（n_points + 2 维）。
    即使对二值图像，LBP 也能编码局部边缘方向和邻域关系。
    """
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_points + 2,
        range=(0, n_points + 2),
        density=True
    )
    return hist.astype(np.float32)


def _extract_density_grid(img: np.ndarray, scales: list[int]) -> np.ndarray:
    """
    多尺度像素密度网格。
    对每个 scale，将图像分成 scale×scale 格子，计算每格白色像素占比。
    """
    binary = (img > 127).astype(np.float32)
    h, w = img.shape
    parts = []
    for scale in scales:
        ch = h // scale
        cw = w // scale
        grid = np.zeros(scale * scale, dtype=np.float32)
        for i in range(scale):
            for j in range(scale):
                cell = binary[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
                grid[i * scale + j] = float(np.mean(cell))
        parts.append(grid)
    return np.concatenate(parts)
