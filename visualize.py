"""
visualize.py - 可视化分析脚本

生成以下图表，保存到 output/plots/：
  1. distance_distribution.png  — 各类距离分布 + 拒绝阈值
  2. confusion_matrix.png       — 混淆矩阵热图
  3. class_performance.png      — 每类分类性能柱状图
  4. tsne_features.png          — t-SNE 特征空间可视化
  5. centroid_images.png        — K-Means 质心重建图像

用法：
  python visualize.py                    # 使用默认 config.yaml
  python visualize.py --config my.yaml
"""

import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹出窗口

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path

from cluster_classifier import ClusterClassifier


# ====================================================================== #
#  数据加载                                                                #
# ====================================================================== #

def load_all_images(config: dict) -> dict[str, list[np.ndarray]]:
    root = Path(config['dataset']['root'])
    img_fmt = config['dataset']['image_format']
    result: dict[str, list[np.ndarray]] = {}
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                for p in sorted(cls_dir.glob(img_fmt))]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            result[cls_dir.name] = imgs
    return result


# ====================================================================== #
#  各图表生成函数                                                           #
# ====================================================================== #

def plot_distance_distribution(
    classifier: ClusterClassifier,
    all_data: dict,
    config: dict,
    out_dir: Path,
) -> None:
    """
    各类样本到最近质心距离的概率密度分布。
    直观显示正常类与 None 类的距离范围，以及拒绝阈值的位置。
    """
    none_name = config['dataset']['none_class_name']
    class_names = classifier.class_names

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(class_names)))

    for cls, color in zip(class_names, colors):
        imgs = all_data.get(cls, [])
        if not imgs:
            continue
        feats = classifier.extractor.transform(imgs)
        dists = classifier._batch_min_distances(feats)
        ax.hist(dists, bins=50, alpha=0.45, color=color, label=cls, density=True)

    # None 类用红色
    none_imgs = all_data.get(none_name, [])
    if none_imgs:
        feats = classifier.extractor.transform(none_imgs)
        dists = classifier._batch_min_distances(feats)
        ax.hist(dists, bins=50, alpha=0.55, color='crimson',
                label=f'{none_name}（异常）', density=True)

    # 拒绝阈值
    thr = classifier.distance_threshold
    ax.axvline(thr, color='black', linestyle='--', linewidth=2,
               label=f'欧氏距离阈值 = {thr:.3f}')

    ax.set_xlabel('到最近质心的欧氏距离', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('各类别距离分布（正常类 vs None 类）', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'distance_distribution.png', dpi=150)
    plt.close()
    print('  ✓ distance_distribution.png')


def plot_confusion_matrix(
    classifier: ClusterClassifier,
    all_data: dict,
    config: dict,
    out_dir: Path,
) -> None:
    """混淆矩阵热图（显示数量和按列归一化百分比）"""
    none_name = config['dataset']['none_class_name']
    class_names = classifier.class_names
    label_list = class_names + [none_name]
    n = len(label_list)
    label_to_idx = {l: i for i, l in enumerate(label_list)}

    confusion = np.zeros((n, n), dtype=np.int64)
    for true_cls, imgs in all_data.items():
        ti = label_to_idx.get(true_cls)
        if ti is None:
            continue
        for img in imgs:
            pred_cls, _, rejected = classifier.predict_single(img)
            pred_label = none_name if rejected else pred_cls
            pi = label_to_idx.get(pred_label, label_to_idx[none_name])
            confusion[pi][ti] += 1

    # 按列归一化为百分比
    col_sums = confusion.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    conf_pct = confusion / col_sums * 100

    fig, ax = plt.subplots(figsize=(n * 1.4 + 1, n * 1.2 + 1))
    im = ax.imshow(conf_pct, cmap='Blues', vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('预测占真实类的比例 (%)', fontsize=10)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_list, rotation=40, ha='right', fontsize=10)
    ax.set_yticklabels(label_list, fontsize=10)
    ax.set_xlabel('真实类别', fontsize=12)
    ax.set_ylabel('预测类别', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=14)

    for i in range(n):
        for j in range(n):
            pct = conf_pct[i][j]
            cnt = confusion[i][j]
            text_color = 'white' if pct > 55 else 'black'
            ax.text(j, i, f'{cnt}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color=text_color)

    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print('  ✓ confusion_matrix.png')


def plot_class_performance(
    classifier: ClusterClassifier,
    all_data: dict,
    config: dict,
    out_dir: Path,
) -> None:
    """每类正确率、误拒率、误分类率的分组柱状图"""
    none_name = config['dataset']['none_class_name']
    class_names = classifier.class_names

    acc_list, rej_list, mis_list = [], [], []

    for cls in class_names:
        imgs = all_data.get(cls, [])
        correct = false_rej = misclassified = 0
        for img in imgs:
            pred_cls, _, rejected = classifier.predict_single(img)
            if rejected:
                false_rej += 1
            elif pred_cls == cls:
                correct += 1
            else:
                misclassified += 1
        n = len(imgs) or 1
        acc_list.append(correct / n * 100)
        rej_list.append(false_rej / n * 100)
        mis_list.append(misclassified / n * 100)

    # None 类检测率
    none_imgs = all_data.get(none_name, [])
    none_detected = sum(1 for img in none_imgs if classifier.predict_single(img)[2])
    none_rate = none_detected / (len(none_imgs) or 1) * 100

    x = np.arange(len(class_names))
    width = 0.27

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：正常类性能
    b1 = ax1.bar(x - width, acc_list, width, label='正确分类率', color='steelblue')
    b2 = ax1.bar(x, rej_list, width, label='误拒率（误判为 None）', color='tomato')
    b3 = ax1.bar(x + width, mis_list, width, label='误分类率（分到其他正常类）', color='darkorange')

    ax1.set_ylabel('百分比 (%)', fontsize=12)
    ax1.set_title('正常类分类性能', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 118)
    ax1.grid(True, axis='y', alpha=0.3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h >= 0.5:
                ax1.annotate(f'{h:.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords='offset points',
                             ha='center', va='bottom', fontsize=8)

    # 右图：None 检测率
    ax2.bar(['None 检测率'], [none_rate], color='crimson', width=0.4)
    ax2.bar(['漏检率'], [100 - none_rate], color='lightgray', width=0.4)
    ax2.set_ylim(0, 118)
    ax2.set_title(f'None 类检测性能（{none_detected}/{len(none_imgs)}）', fontsize=13)
    ax2.set_ylabel('百分比 (%)', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.annotate(f'{none_rate:.1f}%', xy=(0, none_rate),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color='crimson')

    plt.tight_layout()
    plt.savefig(out_dir / 'class_performance.png', dpi=150)
    plt.close()
    print('  ✓ class_performance.png')


def plot_tsne(
    classifier: ClusterClassifier,
    all_data: dict,
    config: dict,
    out_dir: Path,
    max_per_class: int = 150,
) -> None:
    """
    t-SNE 二维特征空间可视化。
    - 正常类用圆点，颜色区分类别
    - None 类用红色叉
    - K-Means 质心用五角星标出
    """
    from sklearn.manifold import TSNE

    none_name = config['dataset']['none_class_name']
    class_names = classifier.class_names
    all_classes_ordered = class_names + [none_name]

    all_feats_list: list[np.ndarray] = []
    all_labels: list[str] = []
    centroid_feats: list[np.ndarray] = []
    centroid_labels: list[str] = []

    for cls in all_classes_ordered:
        imgs = all_data.get(cls, [])
        if not imgs:
            continue
        sampled = imgs[:max_per_class]
        feats = classifier.extractor.transform(sampled)
        all_feats_list.append(feats)
        all_labels.extend([cls] * len(sampled))

    # 加入质心（也参与 t-SNE 拟合）
    k_per_class = len(classifier.all_centroids) // len(class_names)
    for ci, cls in enumerate(class_names):
        start = ci * k_per_class
        end = start + k_per_class
        for c in classifier.all_centroids[start:end]:
            centroid_feats.append(c)
            centroid_labels.append(cls)

    all_feats_arr = np.vstack(all_feats_list)
    centroid_arr = np.array(centroid_feats)
    combined = np.vstack([all_feats_arr, centroid_arr])

    print('  正在计算 t-SNE...')
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1),
                n_iter=1000, verbose=0)
    embedded = tsne.fit_transform(combined)

    n_samples = len(all_feats_arr)
    sample_emb = embedded[:n_samples]
    centroid_emb = embedded[n_samples:]

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(class_names)))
    color_map = dict(zip(class_names, colors))

    fig, ax = plt.subplots(figsize=(13, 10))
    label_arr = np.array(all_labels)

    # 正常类
    for cls, color in zip(class_names, colors):
        mask = label_arr == cls
        ax.scatter(sample_emb[mask, 0], sample_emb[mask, 1],
                   c=[color], label=cls, alpha=0.55, s=25, marker='o')

    # None 类
    none_mask = label_arr == none_name
    if none_mask.any():
        ax.scatter(sample_emb[none_mask, 0], sample_emb[none_mask, 1],
                   c='crimson', label=f'{none_name}（异常）',
                   alpha=0.6, s=30, marker='x', linewidths=1.2)

    # 质心
    c_label_arr = np.array(centroid_labels)
    for cls, color in zip(class_names, colors):
        mask = c_label_arr == cls
        ax.scatter(centroid_emb[mask, 0], centroid_emb[mask, 1],
                   c=[color], s=120, marker='*', edgecolors='black',
                   linewidths=0.8, zorder=5)

    # 图例说明质心
    from matplotlib.lines import Line2D
    legend_extra = [Line2D([0], [0], marker='*', color='gray', markersize=11,
                            markeredgecolor='black', linestyle='None', label='K-Means 质心')]
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extra, lbls + ['K-Means 质心'],
              loc='best', fontsize=10, markerscale=1.3)

    ax.set_title('t-SNE 特征空间可视化', fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('t-SNE 维度 1', fontsize=11)
    ax.set_ylabel('t-SNE 维度 2', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_dir / 'tsne_features.png', dpi=150)
    plt.close()
    print('  ✓ tsne_features.png')


def plot_centroid_images(
    classifier: ClusterClassifier,
    config: dict,
    out_dir: Path,
) -> None:
    """
    将 K-Means 质心反变换回 32x32 像素空间，可视化每类的代表性模式。
    需要 PCA 启用。
    """
    if not config['features']['pca']['enabled']:
        print('  ⚠ PCA 未启用，跳过质心图像重建')
        return

    extractor = classifier.extractor
    class_names = classifier.class_names
    n_classes = len(class_names)
    k_total = len(classifier.all_centroids)
    k_per_class = k_total // n_classes

    fig, axes = plt.subplots(
        n_classes, k_per_class,
        figsize=(k_per_class * 1.6, n_classes * 1.8),
        squeeze=False,
    )
    fig.suptitle('K-Means 质心重建图像（每列为一个聚类中心）', fontsize=13, y=1.01)

    n_pca = int(extractor.pca.n_components_)

    for cls_idx, cls in enumerate(class_names):
        start = cls_idx * k_per_class
        end = start + k_per_class
        centroids_norm = classifier.all_centroids[start:end]  # (k, D) 已归一化

        for k_idx, centroid_norm in enumerate(centroids_norm):
            ax = axes[cls_idx][k_idx]

            # 反 StandardScaler
            raw = extractor.scaler.inverse_transform(centroid_norm.reshape(1, -1))[0]

            # 前 n_pca 维是 PCA 成分，其余是非PCA特征
            pca_part = raw[:n_pca].reshape(1, -1)

            # 反 PCA → 原始像素空间
            pixel_flat = extractor.pca.inverse_transform(pca_part)[0]
            pixel_img = np.clip(pixel_flat * 255, 0, 255).reshape(32, 32)

            ax.imshow(pixel_img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.axis('off')
            if k_idx == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=0, ha='right',
                              va='center', labelpad=40)

    plt.tight_layout()
    plt.savefig(out_dir / 'centroid_images.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ centroid_images.png')


def plot_mahal_distribution(
    classifier: ClusterClassifier,
    all_data: dict,
    config: dict,
    out_dir: Path,
) -> None:
    """
    马氏距离分布图（仅当马氏距离启用时生成）。
    每个子图对应一个正常类：展示该类样本的马氏距离分布 + 阈值。
    """
    if not config['rejection']['mahalanobis']['enabled']:
        return
    if not classifier.mahal_thresholds:
        return

    none_name = config['dataset']['none_class_name']
    class_names = classifier.class_names
    n = len(class_names)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for cls, ax in zip(class_names, axes):
        # 正常类
        imgs = all_data.get(cls, [])
        if imgs:
            feats = classifier.extractor.transform(imgs)
            mdists = classifier._batch_mahal_distances(feats, cls)
            ax.hist(mdists, bins=40, alpha=0.6, color='steelblue',
                    label=cls, density=True)

        # None 类马氏距离（到该正常类）
        none_imgs = all_data.get(none_name, [])
        if none_imgs:
            feats_none = classifier.extractor.transform(none_imgs)
            mdists_none = classifier._batch_mahal_distances(feats_none, cls)
            ax.hist(mdists_none, bins=40, alpha=0.5, color='crimson',
                    label=none_name, density=True)

        # 阈值线
        thr = classifier.mahal_thresholds.get(cls)
        if thr is not None:
            ax.axvline(thr, color='black', linestyle='--', linewidth=1.5,
                       label=f'阈值={thr:.1f}')

        ax.set_title(cls, fontsize=10)
        ax.set_xlabel('马氏距离', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('各类别马氏距离分布', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / 'mahal_distribution.png', dpi=150)
    plt.close()
    print('  ✓ mahal_distribution.png')


# ====================================================================== #
#  主入口                                                                  #
# ====================================================================== #

def main(config_path: str = 'config.yaml') -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_path = config['output']['model_path']
    plots_dir = Path(config['output'].get('plots_dir', 'output/plots'))
    plots_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('RM-Classifier 可视化')
    print('=' * 55)

    classifier = ClusterClassifier.load(model_path)
    print(f'\n加载数据集...')
    all_data = load_all_images(config)
    total = sum(len(v) for v in all_data.values())
    print(f'共 {total} 张图像，{len(all_data)} 个类别\n')

    print('生成图表:')
    plot_distance_distribution(classifier, all_data, config, plots_dir)
    plot_confusion_matrix(classifier, all_data, config, plots_dir)
    plot_class_performance(classifier, all_data, config, plots_dir)
    plot_mahal_distribution(classifier, all_data, config, plots_dir)
    plot_centroid_images(classifier, config, plots_dir)
    plot_tsne(classifier, all_data, config, plots_dir)

    print(f'\n所有图表已保存到: {plots_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成可视化图表')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    main(args.config)
