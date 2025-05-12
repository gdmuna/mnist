# MNIST手写数字识别项目

这是一个使用PyTorch实现的MNIST手写数字识别项目，采用了现代化的CNN架构和训练策略。

## 项目特点

- 使用深度卷积神经网络(CNN)进行手写数字识别
- 实现了多种现代化训练技术：
  - 批量归一化(Batch Normalization)
  - Dropout正则化
  - 数据增强
  - 学习率自适应调整
  - L2正则化
- 支持GPU加速训练
- 自动保存最佳模型和最终模型
- 完整的训练和测试流程

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib
- CUDA（可选，用于GPU加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
best/
├── mnist_best.py    # 主程序文件
├── model_save/      # 模型保存目录
│   ├── best_model.pth    # 最佳模型
│   └── final_model.pth   # 最终模型
└── README.md        # 项目说明文档
```

## 使用方法

1. 运行训练程序：
```bash
python mnist_best.py
```

2. 程序会自动：
   - 下载MNIST数据集
   - 创建必要的目录
   - 开始训练过程
   - 保存最佳模型和最终模型

## 模型架构

该模型采用了三层卷积神经网络结构：

1. 卷积层1：64个3x3卷积核
2. 卷积层2：128个3x3卷积核
3. 卷积层3：256个3x3卷积核
4. 全连接层：1024 -> 512 -> 10

每层卷积后都包含：
- 批量归一化
- ReLU激活函数
- 最大池化

## 训练策略

- 批次大小：128
- 训练轮数：20
- 初始学习率：0.001
- 优化器：Adam
- 学习率调整：ReduceLROnPlateau
- 数据增强：随机旋转和平移
- 正则化：Dropout + L2正则化

## 性能指标

模型在MNIST测试集上可以达到99%以上的准确率。

## 注意事项

1. 首次运行时会自动下载MNIST数据集
2. 如果有GPU，程序会自动使用GPU进行训练
3. 训练过程中会自动保存最佳模型
4. 可以通过修改超参数来调整模型性能

## 许可证

MIT License 