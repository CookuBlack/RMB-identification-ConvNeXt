import cv2
import onnxruntime as ort
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# 配置参数
ONNX_MODEL_PATH = "./output/convnext_tiny_rmb.onnx"  # 导出的ONNX模型路径
CLASS_NAMES = ["1", "10", "100", "20", "5", "50"]  # 类别名称（和训练一致）
INPUT_SIZE = 224  # 模型输入尺寸
# 归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# 推理设备：CPUExecutionProvider / CUDAExecutionProvider
DEVICE_PROVIDER = "CPUExecutionProvider"


# 加载ONNX模型
def load_onnx_session():
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options,
            providers=[DEVICE_PROVIDER]
        )
        input_info = session.get_inputs()[0]
        print(f"成功加载ONNX模型：{ONNX_MODEL_PATH}")
        print(f"推理设备：{DEVICE_PROVIDER}")
        print(f"模型输入要求：形状={input_info.shape}，类型={input_info.type}")
        return session
    except Exception as e:
        raise RuntimeError(f"加载ONNX模型失败：{str(e)}")


# ===================== 图片预处理 =====================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}")

    # BGR转RGB + 强制float32
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # 缩放 + 归一化
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)) / 255.0
    # 标准化
    img = (img - MEAN) / STD
    # 转NCHW格式 + 增加batch维度
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    print(f"预处理后数据类型：{img.dtype}")
    print(f"预处理后数据形状：{img.shape}")
    return img


# 单张图片推理
def infer_single_image(img_path, session):
    input_data = preprocess_image(img_path)
    input_data = input_data.astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    try:
        outputs = session.run([output_name], {input_name: input_data})[0]
        # 计算置信度
        probabilities = np.exp(outputs).astype(np.float32) / np.sum(np.exp(outputs).astype(np.float32), axis=1)
        pred_idx = np.argmax(probabilities, axis=1)[0]
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = round(float(probabilities[0][pred_idx]), 4)

        return pred_class, pred_conf
    except Exception as e:
        raise RuntimeError(f"推理失败：{str(e)}")


# 批量推理
def infer_batch_images(test_dir, session):
    print("\n批量推理结果")
    total = 0
    correct = 0

    for true_class in CLASS_NAMES:
        class_dir = os.path.join(test_dir, true_class)
        if not os.path.exists(class_dir):
            print(f"未找到类别文件夹：{class_dir}")
            continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_name)
            try:
                pred_class, pred_conf = infer_single_image(img_path, session)
                total += 1
                if pred_class == true_class:
                    correct += 1
                print(f"图片：{img_name} | 真实面值：{true_class}元 | 预测面值：{pred_class}元 | 置信度：{pred_conf}")
            except Exception as e:
                print(f"图片{img_name}推理出错：{str(e)}")
                total += 1

    if total > 0:
        accuracy = correct / total
        print(f"\n批量推理完成 | 总样本数：{total} | 正确数：{correct} | 准确率：{accuracy:.4f}")
    else:
        print("\n未找到有效测试图片")


# 运行入口
if __name__ == "__main__":
    # 加载模型
    onnx_session = load_onnx_session()

    # 单张图片推理
    test_img_path = "./inference/100yuan.jpg"
    if os.path.exists(test_img_path):
        pred_class, pred_conf = infer_single_image(test_img_path, onnx_session)
        print("\n推理结果：")
        print(f"图片路径：{test_img_path}")
        print(f"预测面值：{pred_class}元")
        print(f"置信度：{pred_conf}")
    else:
        print(f"测试图片不存在：{test_img_path}")

    # 批量推理
    # test_dir = "./rmb_dataset_split/test"
    # infer_batch_images(test_dir, onnx_session)