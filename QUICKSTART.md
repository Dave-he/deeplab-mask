# 快速开始指南

欢迎使用DeepLab v3+图像分割工具！这个指南将帮助您快速上手。

## 🚀 快速开始

### 1. 环境检查

确保您的系统已安装Python 3.7+：

```bash
python3 --version
```

### 2. 安装依赖

```bash
# 方法1: 使用Makefile（推荐）
make install

# 方法2: 手动安装
pip3 install -r requirements.txt
```

### 3. 生成测试数据

```bash
# 生成示例图像用于测试
python3 generate_sample_data.py --output_dir ./data/input --num_images 5
```

### 4. 运行演示

#### 单张图像演示

```bash
# 使用生成的测试图像
python3 demo.py --image ./data/input/object_scene_001.png --output ./demo_output
```

#### 批量处理演示

```bash
# 批量处理所有测试图像
python3 predict.py --input_dir ./data/input --output_dir ./data/output
```

## 📁 输出结果

处理完成后，您将在输出目录中看到：

```
output_directory/
├── masks/              # 每个类别的二值mask图
│   ├── image_person.png
│   ├── image_car.png
│   └── ...
├── colored_masks/      # 彩色整体mask图
│   └── image_colored.png
└── overlays/          # 原图与mask叠加的可视化
    └── image_overlay.png
```

## 🛠️ 使用自己的图像

### 单张图像

```bash
python3 demo.py --image /path/to/your/image.jpg --output ./my_output
```

### 批量处理

```bash
python3 predict.py --input_dir /path/to/your/images --output_dir ./my_output
```

## ⚙️ 高级选项

### 指定设备

```bash
# 强制使用CPU
python3 predict.py --input_dir ./data/input --output_dir ./data/output --device cpu

# 使用GPU（如果可用）
python3 predict.py --input_dir ./data/input --output_dir ./data/output --device cuda
```

### 调整图像尺寸

```bash
# 将输入图像调整为512x512进行处理
python3 predict.py --input_dir ./data/input --output_dir ./data/output --target_size 512 512
```

### 选择编码器

```bash
# 使用ResNet50编码器（更快但精度稍低）
python3 predict.py --input_dir ./data/input --output_dir ./data/output --encoder resnet50

# 使用ResNet101编码器（默认，精度更高）
python3 predict.py --input_dir ./data/input --output_dir ./data/output --encoder resnet101
```

## 🧪 测试模型

运行完整的模型测试：

```bash
python3 test_model.py
```

这将测试模型的各个功能模块并生成测试报告。

## 📊 支持的类别

模型默认支持PASCAL VOC数据集的21个类别：

1. background（背景）
2. aeroplane（飞机）
3. bicycle（自行车）
4. bird（鸟）
5. boat（船）
6. bottle（瓶子）
7. bus（公交车）
8. car（汽车）
9. cat（猫）
10. chair（椅子）
11. cow（牛）
12. diningtable（餐桌）
13. dog（狗）
14. horse（马）
15. motorbike（摩托车）
16. person（人）
17. pottedplant（盆栽）
18. sheep（羊）
19. sofa（沙发）
20. train（火车）
21. tvmonitor（电视）

## 🔧 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 使用较小的图像尺寸
   python3 predict.py --input_dir ./data/input --output_dir ./data/output --target_size 256 256
   ```

2. **CUDA错误**
   ```bash
   # 强制使用CPU
   python3 predict.py --input_dir ./data/input --output_dir ./data/output --device cpu
   ```

3. **依赖包错误**
   ```bash
   # 重新安装依赖
   pip3 install -r requirements.txt --force-reinstall
   ```

### 获取帮助

```bash
# 查看详细帮助
python3 predict.py --help
python3 demo.py --help

# 查看Makefile命令
make help
```

## 🎯 下一步

- 尝试使用自己的图像数据
- 调整模型参数以获得更好的结果
- 探索不同的编码器选项
- 查看生成的可视化结果

祝您使用愉快！🎉