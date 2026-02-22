import cv2
import torch
from torchvision import transforms
import os
import warnings
from model import convnext_tiny

warnings.filterwarnings("ignore")

# 核心配置
MODEL_WEIGHTS_PATH = "./output/best_model.pth"
CLASS_NAMES = ["1", "10", "100", "20", "5", "50"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载模型
def load_rmb_model():
    model = convnext_tiny(num_classes=6)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        print(f"成功加载模型权重：{MODEL_WEIGHTS_PATH}")
    else:
        raise FileNotFoundError(f"模型权重文件不存在：{MODEL_WEIGHTS_PATH}")

    model = model.to(DEVICE)
    model.eval()
    return model


# 图片预处理
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片：{img_path}（路径错误/图片损坏）")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    return input_tensor


# 单张图片推理
def predict_rmb_denomination(img_path, model):
    input_tensor = preprocess_image(img_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = round(probabilities[0][pred_idx].item(), 4)

    return pred_class, pred_conf


# 批量推理
def batch_predict_rmb(test_dir, model):
    print("\n批量推理结果")
    total = 0
    correct = 0

    for true_class in CLASS_NAMES:
        class_dir = os.path.join(test_dir, true_class)
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_name)
            pred_class, pred_conf = predict_rmb_denomination(img_path, model)

            total += 1
            if pred_class == true_class:
                correct += 1

            print(f"图片：{img_name} | 真实面值：{true_class}元 | 预测面值：{pred_class}元 | 置信度：{pred_conf}")

    if total > 0:
        accuracy = correct / total
        print(f"\n批量推理完成 | 总样本数：{total} | 正确数：{correct} | 准确率：{accuracy:.4f}")
    else:
        print("\n未找到有效测试图片")


# 运行入口
if __name__ == "__main__":
    model = load_rmb_model()

    # 单张图片推理
    test_img_path = "./inference/5yuan.png"
    if os.path.exists(test_img_path):
        pred_class, pred_conf = predict_rmb_denomination(test_img_path, model)
        print("\n推理结果：")
        print(f"图片路径：{test_img_path}")
        print(f"预测面值：{pred_class}元")
        print(f"置信度：{pred_conf}")
    else:
        print(f"测试图片不存在：{test_img_path}")

    # 批量推理
    # test_dir = "./rmb_dataset_split/test"
    # batch_predict_rmb(test_dir, model)