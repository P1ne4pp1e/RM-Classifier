import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path
import time


class ONNXClassifier:
    """
    ONNX模型推理类
    支持单张图片和批量图片推理
    """

    def __init__(self, model_path, class_names=None, input_size=(32, 32)):
        """
        初始化推理器

        参数:
            model_path: ONNX模型路径
            class_names: 类别名称列表
            input_size: 输入图片尺寸 (height, width)
        """
        self.model_path = model_path
        self.input_size = input_size

        # 默认类别名称
        if class_names is None:
            self.class_names = [f'collection{i}' for i in range(1, 9)] + ['collection-none']
        else:
            self.class_names = class_names

        # 图像预处理参数（与训练时一致）
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 加载ONNX模型
        self._load_model()

        print(f"✅ ONNX推理器初始化完成")
        print(f"   模型: {Path(model_path).name}")
        print(f"   类别数: {len(self.class_names)}")
        print(f"   输入尺寸: {input_size}")

    def _load_model(self):
        """加载ONNX模型"""
        # 设置推理选项（使用所有可用CPU核心）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # 自动使用所有CPU核心

        # 创建推理会话
        providers = ['CPUExecutionProvider']

        # 如果有GPU，优先使用GPU
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            print("✅ 检测到CUDA，使用GPU推理")

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image):
        """
        图像预处理

        参数:
            image: PIL.Image对象或numpy数组

        返回:
            preprocessed: 预处理后的numpy数组 (1, 3, H, W)
        """
        # 转换为PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize
        image = image.resize(self.input_size, Image.BILINEAR)

        # 转为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 转为numpy数组并归一化到[0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # 标准化
        img_array = (img_array - self.mean) / self.std

        # 转换维度: (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))

        # 添加batch维度: (C, H, W) -> (1, C, H, W)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image, return_probs=False):
        """
        单张图片推理

        参数:
            image: PIL.Image对象、numpy数组或图片路径
            return_probs: 是否返回概率分布

        返回:
            如果return_probs=False: (class_id, class_name, confidence)
            如果return_probs=True: (class_id, class_name, confidence, probs)
        """
        # 如果是路径，先加载图片
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # 预处理
        input_data = self.preprocess(image)

        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})[0]

        # Softmax获取概率
        probs = self._softmax(outputs[0])

        # 获取预测结果
        class_id = int(np.argmax(probs))
        class_name = self.class_names[class_id]
        confidence = float(probs[class_id])

        if return_probs:
            return class_id, class_name, confidence, probs
        else:
            return class_id, class_name, confidence

    def predict_batch(self, images, return_probs=False):
        """
        批量推理

        参数:
            images: 图片列表（PIL.Image、numpy数组或路径）
            return_probs: 是否返回概率分布

        返回:
            results: 推理结果列表
        """
        # 预处理所有图片
        batch_data = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            batch_data.append(self.preprocess(img))

        # 合并为batch
        batch_data = np.concatenate(batch_data, axis=0)

        # 批量推理
        outputs = self.session.run([self.output_name], {self.input_name: batch_data})[0]

        # 处理结果
        results = []
        for output in outputs:
            probs = self._softmax(output)
            class_id = int(np.argmax(probs))
            class_name = self.class_names[class_id]
            confidence = float(probs[class_id])

            if return_probs:
                results.append((class_id, class_name, confidence, probs))
            else:
                results.append((class_id, class_name, confidence))

        return results

    def benchmark(self, num_runs=1000):
        """
        性能测试

        参数:
            num_runs: 测试次数

        返回:
            avg_time: 平均推理时间（毫秒）
        """
        # 创建随机输入
        dummy_input = np.random.randn(1, 3, *self.input_size).astype(np.float32)

        # 预热
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: dummy_input})

        # 计时
        start = time.time()
        for _ in range(num_runs):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        end = time.time()

        avg_time = (end - start) / num_runs * 1000

        print(f"\n{'=' * 50}")
        print(f"性能测试结果 ({num_runs}次推理)")
        print(f"{'=' * 50}")
        print(f"平均推理时间: {avg_time:.3f} ms")
        print(f"FPS: {1000 / avg_time:.1f}")
        print(f"{'=' * 50}")

        return avg_time

    @staticmethod
    def _softmax(x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def print_prediction(self, class_id, class_name, confidence, probs=None):
        """打印预测结果（格式化输出）"""
        print(f"\n{'=' * 50}")
        print(f"预测结果:")
        print(f"{'=' * 50}")
        print(f"类别ID: {class_id}")
        print(f"类别名称: {class_name}")
        print(f"置信度: {confidence:.2%}")

        if probs is not None:
            print(f"\n所有类别概率:")
            for i, (name, prob) in enumerate(zip(self.class_names, probs)):
                bar = '█' * int(prob * 50)
                print(f"  {name:20s} {prob:.2%} {bar}")
        print(f"{'=' * 50}")


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 1. 初始化推理器
    classifier = ONNXClassifier(
        model_path='./output/mlp_overfit_01.onnx',
        class_names=[f'collection{i}' for i in range(1, 9)] + ['collection-none'],
        input_size=(32, 32)
    )

    # 2. 单张图片推理
    print("\n【示例1: 单张图片推理】")
    image_path = './test_image.jpg'  # 替换为你的图片路径

    try:
        class_id, class_name, confidence, probs = classifier.predict(
            image_path,
            return_probs=True
        )
        classifier.print_prediction(class_id, class_name, confidence, probs)
    except FileNotFoundError:
        print(f"⚠️  图片不存在: {image_path}")

    # 3. 批量推理
    print("\n【示例2: 批量推理】")
    image_folder = Path('./dataset/collection1')  # 替换为你的图片文件夹

    if image_folder.exists():
        image_paths = list(image_folder.glob('*.jpg'))[:5]  # 取前5张

        if image_paths:
            results = classifier.predict_batch(image_paths)

            print(f"\n批量推理结果:")
            for i, (img_path, (cls_id, cls_name, conf)) in enumerate(zip(image_paths, results)):
                print(f"  {i + 1}. {img_path.name:30s} -> {cls_name:20s} ({conf:.2%})")
        else:
            print("⚠️  文件夹中没有jpg图片")
    else:
        print(f"⚠️  文件夹不存在: {image_folder}")

    # 4. 性能测试
    print("\n【示例3: 性能测试】")
    classifier.benchmark(num_runs=1000)

    # 5. 使用PIL Image对象推理
    print("\n【示例4: 使用PIL Image对象】")
    from PIL import Image

    # 创建一个随机图片用于演示
    random_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    class_id, class_name, confidence = classifier.predict(random_img)
    print(f"随机图片预测: {class_name} ({confidence:.2%})")