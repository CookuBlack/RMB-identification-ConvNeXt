import os
import shutil
from sklearn.model_selection import train_test_split

# 原始数据集根目录
source_dir = r"RMBDataset"

# 划分后数据集的保存路径
target_dir = r"./rmb_dataset_split"
train_ratio = 0.8   # 训练集占比

os.makedirs(target_dir, exist_ok=True)
# 创建train和test子目录
train_root = os.path.join(target_dir, "train")
test_root = os.path.join(target_dir, "test")
os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

# 遍历每个面值文件夹
for class_name in os.listdir(source_dir):
    # 拼接当前面值的完整路径
    class_path = os.path.join(source_dir, class_name)
    # 跳过非文件夹
    if not os.path.isdir(class_path):
        continue

    # 获取当前面值下的所有图片
    img_list = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(img_list) == 0:
        print(f"面值{class_name}文件夹为空")
        continue

    # 随机划分训练集和测试集
    train_imgs, test_imgs = train_test_split(
        img_list,
        train_size=train_ratio,
        random_state=42,    # 固定种子，每次划分结果一样
        shuffle=True        # 打乱图片顺序
    )

    # 复制训练集图片
    train_class_dir = os.path.join(train_root, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    for img in train_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy(src, dst)

    # 复制测试集图片
    test_class_dir = os.path.join(test_root, class_name)
    os.makedirs(test_class_dir, exist_ok=True)
    for img in test_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_class_dir, img)
        shutil.copy(src, dst)

    # 打印划分结果，确认是否成功
    print(f"面值{class_name}：训练集{len(train_imgs)}张 | 测试集{len(test_imgs)}张")

print(f"划分完成！新数据集路径：{target_dir}")