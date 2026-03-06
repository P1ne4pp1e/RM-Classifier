"""
export_model.py - 导出模型参数供 C++ 推理使用

将 joblib 模型中所有推理所需的参数（PCA、Scaler、质心、马氏距离矩阵等）
导出为 OpenCV FileStorage YAML 格式，C++ 端通过 cv::FileStorage 直接加载。

输出文件路径：与 model_path 相同，后缀改为 .yaml
  例：output/cluster_model.joblib → output/cluster_model.yaml

用法：
  python export_model.py                    # 使用默认 config.yaml
  python export_model.py --config my.yaml  # 指定配置文件
"""

import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path

from cluster_classifier import ClusterClassifier


def export_to_yaml(classifier: ClusterClassifier, config: dict, output_path: str) -> None:
    n_classes  = len(classifier.class_names)
    k_total    = len(classifier.all_centroids)
    k_per_cls  = k_total // n_classes
    n_pca      = int(classifier.extractor.pca.n_components_) if classifier.extractor.pca else 0
    feat_dim   = classifier.all_centroids.shape[1]
    n_desc     = config['features']['fourier_descriptors']['n_descriptors']
    mahal_on   = config['rejection']['mahalanobis']['enabled']

    fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)

    # ── 元信息 ────────────────────────────────────────────────────────────
    fs.write("n_classes",        int(n_classes))
    fs.write("k_per_class",      int(k_per_cls))
    fs.write("n_pca_components", int(n_pca))
    fs.write("n_descriptors",    int(n_desc))
    fs.write("feature_dim",      int(feat_dim))
    fs.write("distance_threshold", float(classifier.distance_threshold))
    fs.write("mahal_enabled",    int(mahal_on))
    fs.write("none_class",       classifier.none_class)
    fs.write("class_names_csv",  ";".join(classifier.class_names))

    # 特征开关（C++ 端按此决定是否提取对应特征）
    fs.write("hu_moments_enabled", int(config['features']['hu_moments']['enabled']))
    fs.write("fourier_enabled",    int(config['features']['fourier_descriptors']['enabled']))
    fs.write("topology_enabled",   int(config['features']['topology']['enabled']))

    # ── 马氏距离阈值（按类别顺序存为行向量）───────────────────────────────
    if mahal_on and classifier.mahal_thresholds:
        thresholds = np.array(
            [classifier.mahal_thresholds.get(cls, 0.0) for cls in classifier.class_names],
            dtype=np.float32
        ).reshape(1, -1)
        fs.write("mahal_thresholds", thresholds)

    # ── PCA 参数 ──────────────────────────────────────────────────────────
    if classifier.extractor.pca is not None:
        pca_mean = classifier.extractor.pca.mean_.astype(np.float32).reshape(1, -1)   # (1, 1024)
        pca_comp = classifier.extractor.pca.components_.astype(np.float32)            # (n_pca, 1024)
        fs.write("pca_mean",       pca_mean)
        fs.write("pca_components", pca_comp)

    # ── StandardScaler 参数 ───────────────────────────────────────────────
    scaler_mean  = classifier.extractor.scaler.mean_.astype(np.float32).reshape(1, -1)
    scaler_scale = classifier.extractor.scaler.scale_.astype(np.float32).reshape(1, -1)
    fs.write("scaler_mean",  scaler_mean)
    fs.write("scaler_scale", scaler_scale)

    # ── K-Means 质心 ──────────────────────────────────────────────────────
    centroids = classifier.all_centroids.astype(np.float32)                     # (K, D)
    labels    = classifier.centroid_labels.astype(np.int32).reshape(1, -1)      # (1, K)
    fs.write("all_centroids",   centroids)
    fs.write("centroid_labels", labels)

    # ── 每类马氏距离参数 ──────────────────────────────────────────────────
    if mahal_on:
        for i, cls in enumerate(classifier.class_names):
            if cls in classifier.class_means:
                mu      = classifier.class_means[cls].astype(np.float32).reshape(1, -1)
                cov_inv = classifier.class_cov_inv[cls].astype(np.float32)      # (D, D)
                fs.write(f"class_mean_{i}",    mu)
                fs.write(f"class_cov_inv_{i}", cov_inv)

    fs.release()

    print(f"  类别: {classifier.class_names}")
    print(f"  K/类={k_per_cls}, 特征维度={feat_dim}（PCA={n_pca}）")
    print(f"  马氏距离: {'启用' if mahal_on else '禁用'}")
    print(f"  模型已导出 → {output_path}")


def main(config_path: str = 'config.yaml') -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_path  = config['output']['model_path']
    output_path = str(Path(model_path).with_suffix('.yaml'))

    print("=" * 55)
    print("RM-Classifier 模型导出（→ C++ YAML）")
    print("=" * 55)
    print(f"\n加载模型: {model_path}")

    classifier = ClusterClassifier.load(model_path)
    export_to_yaml(classifier, config, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出模型为 C++ 可读的 YAML 格式')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    main(args.config)
