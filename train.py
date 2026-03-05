"""
train.py - 训练入口

用法：
  conda activate pip-classifier
  python train.py                   # 使用默认 config.yaml
  python train.py --config my.yaml  # 指定配置文件
"""

import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

from features import FeatureExtractor
from cluster_classifier import ClusterClassifier


def load_dataset(config: dict) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    """
    按文件夹加载数据集。

    Returns:
        normal_data: {class_name: [img, ...]}
        none_data:   [img, ...]
    """
    root = Path(config['dataset']['root'])
    none_name = config['dataset']['none_class_name']
    img_fmt = config['dataset']['image_format']
    verbose = config['output']['verbose']

    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root.resolve()}")

    normal_data: dict[str, list[np.ndarray]] = {}
    none_data: list[np.ndarray] = []

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue

        imgs = []
        for img_path in sorted(cls_dir.glob(img_fmt)):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)

        if not imgs:
            continue

        if cls_dir.name == none_name:
            none_data = imgs
        else:
            normal_data[cls_dir.name] = imgs

        if verbose:
            label = "(None类)" if cls_dir.name == none_name else ""
            print(f"  {cls_dir.name}: {len(imgs)} 张 {label}")

    return normal_data, none_data


def split_normal_data(
    normal_data: dict[str, list[np.ndarray]],
    val_split: float,
    seed: int,
) -> tuple[dict, dict]:
    """将每个正常类按比例划分训练/验证集"""
    train_data: dict[str, list[np.ndarray]] = {}
    val_data: dict[str, list[np.ndarray]] = {}

    for cls, imgs in normal_data.items():
        indices = list(range(len(imgs)))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=seed
        )
        train_data[cls] = [imgs[i] for i in train_idx]
        val_data[cls] = [imgs[i] for i in val_idx]

    return train_data, val_data


def evaluate_validation(
    classifier: ClusterClassifier,
    val_data: dict[str, list[np.ndarray]],
    none_val: list[np.ndarray],
    config: dict,
) -> None:
    """在验证集上打印分类性能摘要"""
    none_name = config['dataset']['none_class_name']

    print(f"\n  {'类别':<22} {'正确率':>8} {'误拒率':>8} {'样本数':>8}")
    print(f"  {'-' * 52}")

    total_correct = total_false_rej = total_n = 0

    for cls, imgs in sorted(val_data.items()):
        correct = false_rej = 0
        for img in imgs:
            pred_cls, _, rejected = classifier.predict_single(img)
            if rejected:
                false_rej += 1
            elif pred_cls == cls:
                correct += 1
        n = len(imgs)
        acc = correct / n * 100
        rej = false_rej / n * 100
        print(f"  {cls:<22} {acc:>7.1f}% {rej:>7.1f}% {n:>8}")
        total_correct += correct
        total_false_rej += false_rej
        total_n += n

    overall_acc = total_correct / total_n * 100 if total_n else 0
    overall_rej = total_false_rej / total_n * 100 if total_n else 0
    print(f"  {'总体':<22} {overall_acc:>7.1f}% {overall_rej:>7.1f}% {total_n:>8}")

    if none_val:
        rejected_cnt = sum(1 for img in none_val if classifier.predict_single(img)[2])
        rate = rejected_cnt / len(none_val) * 100
        print(f"\n  None 检测率（验证集）: {rate:.1f}%  "
              f"({rejected_cnt}/{len(none_val)})")


def main(config_path: str = 'config.yaml') -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    verbose = config['output']['verbose']

    print("=" * 55)
    print("RM-Classifier 训练")
    print("=" * 55)

    # ── 加载数据 ────────────────────────────────────────────────────────
    if verbose:
        print("\n[数据集] 加载中...")
    normal_data, none_data = load_dataset(config)

    print(f"\n正常类: {sorted(normal_data.keys())}")
    print(f"None 类: {len(none_data)} 张")

    # ── 划分训练/验证集 ─────────────────────────────────────────────────
    val_split = config['dataset']['val_split']
    seed = config['dataset']['random_seed']
    train_data, val_data = split_normal_data(normal_data, val_split, seed)

    if verbose:
        print("\n训练 / 验证集划分:")
        for cls in sorted(train_data):
            print(f"  {cls}: {len(train_data[cls])} 训练 / {len(val_data[cls])} 验证")

    # None 类也划分一个验证子集用于最终报告
    none_val: list[np.ndarray] = []
    none_train: list[np.ndarray] = []
    if none_data:
        split_pt = int(len(none_data) * (1 - val_split))
        none_train = none_data[:split_pt]
        none_val = none_data[split_pt:]

    # ── 拟合特征提取器（仅用训练集）────────────────────────────────────
    all_train_imgs: list[np.ndarray] = []
    for imgs in train_data.values():
        all_train_imgs.extend(imgs)

    if verbose:
        print(f"\n[特征] 拟合特征提取器（{len(all_train_imgs)} 张正常类训练图像）...")
    extractor = FeatureExtractor(config['features'])
    extractor.fit(all_train_imgs, verbose=verbose)

    # ── 训练分类器 ──────────────────────────────────────────────────────
    if verbose:
        print("\n[聚类] 开始训练...")
    classifier = ClusterClassifier(config, extractor)
    # 把验证集传入，让阈值在验证集（unseen data）上校准，避免训练集自循环偏差
    classifier.train(
        train_data,
        none_train if none_train else None,
        val_data=val_data,
    )

    # ── 验证集评估 ──────────────────────────────────────────────────────
    if verbose:
        print("\n[评估] 验证集性能:")
    evaluate_validation(classifier, val_data, none_val, config)

    # ── 保存模型 ────────────────────────────────────────────────────────
    model_path = config['output']['model_path']
    classifier.save(model_path)
    print(f"\n训练完成！模型 → {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练聚类分类器')
    parser.add_argument(
        '--config', default='config.yaml',
        help='配置文件路径（默认：config.yaml）'
    )
    args = parser.parse_args()
    main(args.config)
