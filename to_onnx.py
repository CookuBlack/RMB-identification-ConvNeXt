import torch
import os
import warnings
from model import convnext_tiny

warnings.filterwarnings("ignore")

# 配置参数
MODEL_WEIGHTS_PATH = "./output/best_model.pth"  # 训练好的权重路径
ONNX_SAVE_PATH = "./output/convnext_tiny_rmb.onnx"  # ONNX模型保存路径
INPUT_SIZE = 224    # 模型输入尺寸
NUM_CLASSES = 6     # 类别数


#导出函数
def export_pytorch_to_onnx():
    # 创建输出目录
    os.makedirs(os.path.dirname(ONNX_SAVE_PATH), exist_ok=True)

    # 加载模型
    model = convnext_tiny(num_classes=NUM_CLASSES)
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"模型权重文件不存在：{MODEL_WEIGHTS_PATH}")

    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.float()  # 强制float32，避免推理类型不匹配
    model.eval()  # 切换到推理模式
    print("成功加载PyTorch模型权重")

    # 构造虚拟输入
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=18,
        verbose=False,
        do_constant_folding=True
    )
    print(f"ONNX模型导出完成，保存路径：{ONNX_SAVE_PATH}")


if __name__ == "__main__":
    export_pytorch_to_onnx()