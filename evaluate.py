"""
evaluate.py - 全量评估脚本

功能：
  - 混淆矩阵（正常类 + None 类）
  - 每类准确率、误拒率
  - None 类检测率
  - 推理速度基准测试

用法：
  conda activate pip-classifier
  python evaluate.py                   # 使用默认 config.yaml
  python evaluate.py --config my.yaml  # 指定配置文件
  python evaluate.py --bench 2000      # 指定 benchmark 次数
"""

import argparse
import time
import yaml
import numpy as np
import cv2
from pathlib import Path

from cluster_classifier import ClusterClassifier


def load_all_images(config: dict) -> dict[str, list[np.ndarray]]:
    root = Path(config['dataset']['root'])
    img_fmt = config['dataset']['image_format']
    result: dict[str, list[np.ndarray]] = {}

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = []
        for p in sorted(cls_dir.glob(img_fmt)):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
        if imgs:
            result[cls_dir.name] = imgs

    return result


def run_benchmark(classifier: ClusterClassifier, n_runs: int = 1000) -> float:
    """
    单张图像推理速度基准测试（含特征提取）。
    返回平均推理时间（毫秒）。
    """
    dummy = np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255

    # 预热
    for _ in range(50):
        classifier.predict_single(dummy)

    start = time.perf_counter()
    for _ in range(n_runs):
        classifier.predict_single(dummy)
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000


def main(config_path: str = 'config.yaml', bench_runs: int = 1000) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_path = config['output']['model_path']
    report_path = config['output']['report_path']
    none_name = config['dataset']['none_class_name']

    print("=" * 60)
    print("RM-Classifier 全量评估")
    print("=" * 60)

    # ── 加载模型 ────────────────────────────────────────────────────────
    classifier = ClusterClassifier.load(model_path)
    class_names = classifier.class_names

    # ── 加载全量数据集 ──────────────────────────────────────────────────
    all_data = load_all_images(config)
    print(f"已加载 {sum(len(v) for v in all_data.values())} 张图像")

    # ── 构建混淆矩阵 ────────────────────────────────────────────────────
    # 行 = 预测类别（含 none），列 = 真实类别
    label_list = class_names + [none_name]
    n_labels = len(label_list)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    confusion = np.zeros((n_labels, n_labels), dtype=np.int64)

    total_imgs = 0
    start = time.perf_counter()

    for true_cls, imgs in all_data.items():
        true_idx = label_to_idx.get(true_cls)
        if true_idx is None:
            # 数据集中有未知类别（非配置中的类别）
            continue
        for img in imgs:
            pred_cls, _, rejected = classifier.predict_single(img)
            pred_label = none_name if rejected else pred_cls
            pred_idx = label_to_idx.get(pred_label, label_to_idx[none_name])
            confusion[pred_idx][true_idx] += 1
            total_imgs += 1

    elapsed = time.perf_counter() - start
    avg_ms_inference = elapsed / total_imgs * 1000

    # ── Benchmark（纯推理速度）─────────────────────────────────────────
    print(f"\n[Benchmark] 运行 {bench_runs} 次单张推理...")
    bench_ms = run_benchmark(classifier, bench_runs)

    # ── 组装报告 ────────────────────────────────────────────────────────
    lines: list[str] = []

    def ln(s: str = '') -> None:
        lines.append(s)

    ln("=" * 60)
    ln("RM-Classifier 评估报告")
    ln("=" * 60)
    ln(f"模型文件 : {model_path}")
    ln(f"总评估样本: {total_imgs}")
    ln(f"含特征提取推理速度: {avg_ms_inference:.3f} ms/图像 "
       f"({1000 / avg_ms_inference:.0f} FPS)")
    ln(f"纯推理 Benchmark   : {bench_ms:.3f} ms/图像 "
       f"({1000 / bench_ms:.0f} FPS，{bench_runs} 次平均）")

    # 正常类性能
    ln()
    ln("─── 正常类分类性能 ───")
    ln(f"{'类别':<22} {'正确率':>8} {'误拒率':>8} {'误分率':>8} {'样本数':>8}")
    ln("─" * 58)

    total_correct = total_false_rej = total_misclassified = total_normal = 0

    for cls in class_names:
        ci = label_to_idx[cls]
        col_total = int(confusion[:, ci].sum())
        if col_total == 0:
            continue
        correct = int(confusion[ci][ci])
        false_rej = int(confusion[label_to_idx[none_name]][ci])
        misclassified = col_total - correct - false_rej

        acc = correct / col_total * 100
        rej_rate = false_rej / col_total * 100
        mis_rate = misclassified / col_total * 100

        ln(f"{cls:<22} {acc:>7.1f}% {rej_rate:>7.1f}% {mis_rate:>7.1f}% {col_total:>8}")
        total_correct += correct
        total_false_rej += false_rej
        total_misclassified += misclassified
        total_normal += col_total

    if total_normal:
        overall_acc = total_correct / total_normal * 100
        overall_rej = total_false_rej / total_normal * 100
        overall_mis = total_misclassified / total_normal * 100
        ln("─" * 58)
        ln(f"{'总体':<22} {overall_acc:>7.1f}% {overall_rej:>7.1f}% "
           f"{overall_mis:>7.1f}% {total_normal:>8}")

    # None 类性能
    if none_name in all_data:
        ni = label_to_idx[none_name]
        none_total = int(confusion[:, ni].sum())
        none_detected = int(confusion[ni][ni])
        if none_total:
            detection_rate = none_detected / none_total * 100
            ln()
            ln("─── None 类检测性能 ───")
            ln(f"总 None 样本: {none_total}")
            ln(f"正确拒绝    : {none_detected}  ({detection_rate:.1f}%)")
            ln(f"漏检（误分为正常类）: {none_total - none_detected}  "
               f"({100 - detection_rate:.1f}%)")
            # 漏检细分到每个正常类
            if none_total - none_detected > 0:
                ln("  漏检分布:")
                for cls in class_names:
                    ci = label_to_idx[cls]
                    cnt = int(confusion[ci][ni])
                    if cnt > 0:
                        ln(f"    → 被误判为 {cls}: {cnt}")

    # 混淆矩阵
    ln()
    ln("─── 混淆矩阵（行=预测，列=真实）───")
    col_w = 10
    header_cells = [f"{l[:col_w - 1]:>{col_w}}" for l in label_list]
    ln(f"{'预测 \\ 真实':<22}" + "".join(header_cells))
    ln("─" * (22 + col_w * n_labels))
    for i, pred_label in enumerate(label_list):
        row_cells = [f"{int(confusion[i][j]):>{col_w}}" for j in range(n_labels)]
        ln(f"{pred_label:<22}" + "".join(row_cells))

    report = "\n".join(lines)
    print("\n" + report)

    # 保存报告
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估聚类分类器')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--bench', type=int, default=1000, help='Benchmark 推理次数')
    args = parser.parse_args()
    main(args.config, args.bench)
