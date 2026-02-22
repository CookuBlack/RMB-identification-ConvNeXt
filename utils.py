import matplotlib.pyplot as plt

# 绘制损失图
def loss_chart(train_losses: list[float], val_losses: list[float], epoch: int, loss_plot_path: str) -> None:
    plt.figure(figsize=(10, 6))
    # 绘制训练损失折线
    plt.plot(range(1, epoch + 1), train_losses, label='Train Loss', color='blue', linewidth=2)
    # 绘制验证损失折线
    plt.plot(range(1, epoch + 1), val_losses, label='Val Loss', color='red', linewidth=2)
    # 添加图表样式
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # 保存图片
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # 测试
    loss_chart([1., 2., 3., 3., 1.], [4., 2., 1., 2., 2.], epoch=5, loss_plot_path="./test.png")