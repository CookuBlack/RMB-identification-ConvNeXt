import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os

from model import convnext_tiny
from utils import loss_chart

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./rmb_dataset_split"
NUM_WORKERS = 0  # 多线程读取
INPUT_SIZE = 224  # ConvNeXt标准输入尺寸
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
NUM_CLASSES = 6
SAVE_PATH = "./output/best_model.pth"  # 最优模型保存路径
METRICS_PATH = "./output/metrics.txt"  # 指标/混淆矩阵保存路径
LOSS_PLOT_PATH = "./output/loss_plot.png"   # 损失图保存路径

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),  # 先放大到256×256
    transforms.RandomCrop((INPUT_SIZE, INPUT_SIZE)),        # 随机裁剪回224×224（模拟手在画面不同位置）
    transforms.RandomHorizontalFlip(p=0.5),                 # 随机水平翻转（手势左右对称不影响识别）
    transforms.ColorJitter(
        brightness=0.2,  # 亮度±20%
        contrast=0.2     # 对比度±20%
    ),  # 适配不同光线
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 验证/推理集
val_infer_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# 加载数据集
train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, "train"),  # 训练集路径
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, "test"),  # 验证集路径
    transform=val_infer_transform
)

# 构建DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 训练集必须打乱
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 验证集不用打乱
    num_workers=NUM_WORKERS
)

# 模型/优化器/损失函数
model = convnext_tiny(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


best_f1 = 0.0
# 初始化列表记录每个epoch的损失
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)
    # 记录当前epoch的训练损失
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_loss += criterion(out, labels).item() * imgs.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    # 记录当前epoch的验证损失
    val_losses.append(val_loss)

    # 计算指标
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    # report_text = classification_report(all_labels, all_preds, zero_division=0)  # 文本格式报告
    f1 = report_dict['macro avg']['f1-score']
    acc = report_dict['accuracy']

    # 打印关键信息
    print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # 保存最优模型+指标
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), SAVE_PATH)
        # 写入指标+混淆矩阵到文本
        cm = confusion_matrix(all_labels, all_preds)
        with open(METRICS_PATH, 'w', encoding='utf-8') as f:
            f.write(f"最优Epoch: {epoch+1}\n")
            f.write(f"TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}\n")
            f.write("分类报告:\n" + classification_report(all_labels, all_preds, zero_division=0))
            f.write("\n混淆矩阵（行=真实标签，列=预测标签）:\n")
            f.write(np.array2string(cm, formatter={'int': lambda x: f"{x:4d}"}))


loss_chart(train_losses=train_losses, val_losses=val_losses, epoch=EPOCHS, loss_plot_path=LOSS_PLOT_PATH)
print(f"\n训练完成！最优模型保存至: {SAVE_PATH}，指标保存至: {METRICS_PATH}")