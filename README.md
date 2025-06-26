# DeepLab v3+ 图像分割工具

这个项目使用DeepLab v3+模型对图像进行语义分割，可以批量处理文件夹中的图片并生成不同类型的mask图。

## 功能特点

- 使用预训练的DeepLab v3+模型进行图像分割
- 支持批量处理图像文件夹
- 为每个类别单独生成mask图
- 生成整体的彩色mask图
- 支持自定义模型权重

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 批量预测

```bash
python predict.py --input_dir /path/to/images --output_dir /path/to/output --weights /path/to/model_weights.pth
```

参数说明：
- `--input_dir`: 输入图像文件夹路径
- `--output_dir`: 输出mask图文件夹路径
- `--weights`: 模型权重文件路径（可选，默认使用预训练权重）
- `--device`: 使用的设备（'cuda'或'cpu'，默认自动选择）

## 输出说明

程序会在输出目录下创建以下子文件夹：
- `masks/`: 包含每个类别的二值mask图
- `colored_masks/`: 包含彩色的整体mask图
- `overlays/`: 包含原图与mask叠加的可视化结果