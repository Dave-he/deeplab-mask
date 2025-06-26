#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - DeepLab v3+ 图像分割项目
"""

import os

# 模型配置
MODEL_CONFIG = {
    'num_classes': 21,  # PASCAL VOC 数据集类别数
    'encoder_name': 'resnet101',  # 编码器类型
    'input_size': (512, 512),  # 输入图像尺寸
    'device': 'auto',  # 设备选择
}

# 支持的编码器类型
SUPPORTED_ENCODERS = [
    'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d',
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
    'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
    'mobilenet_v2', 'xception'
]

# PASCAL VOC 类别配置
PASCAL_VOC_CLASSES = {
    'names': [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ],
    'colors': [
        [0, 0, 0],       # background - 黑色
        [128, 0, 0],     # aeroplane - 深红色
        [0, 128, 0],     # bicycle - 深绿色
        [128, 128, 0],   # bird - 橄榄色
        [0, 0, 128],     # boat - 深蓝色
        [128, 0, 128],   # bottle - 紫色
        [0, 128, 128],   # bus - 青色
        [128, 128, 128], # car - 灰色
        [64, 0, 0],      # cat - 栗色
        [192, 0, 0],     # chair - 红色
        [64, 128, 0],    # cow - 黄绿色
        [192, 128, 0],   # diningtable - 橙色
        [64, 0, 128],    # dog - 深紫色
        [192, 0, 128],   # horse - 洋红色
        [64, 128, 128],  # motorbike - 深青色
        [192, 128, 128], # person - 浅灰色
        [0, 64, 0],      # pottedplant - 深绿色
        [128, 64, 0],    # sheep - 棕色
        [0, 192, 0],     # sofa - 亮绿色
        [128, 192, 0],   # train - 黄绿色
        [0, 64, 128]     # tvmonitor - 蓝绿色
    ]
}

# 图像处理配置
IMAGE_CONFIG = {
    'supported_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
    'max_size': (2048, 2048),  # 最大处理尺寸
    'min_size': (32, 32),      # 最小处理尺寸
    'interpolation': 'bilinear',  # 插值方法
}

# 输出配置
OUTPUT_CONFIG = {
    'save_individual_masks': True,   # 保存单独类别mask
    'save_colored_masks': True,      # 保存彩色mask
    'save_overlays': True,           # 保存叠加图像
    'save_visualizations': True,     # 保存可视化结果
    'overlay_alpha': 0.5,           # 叠加透明度
    'dpi': 150,                     # 图像保存DPI
}

# 预训练模型下载链接（示例）
PRETRAINED_MODELS = {
    'pascal_voc': {
        'resnet101': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
        'resnet50': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    },
    'cityscapes': {
        'resnet101': 'path/to/cityscapes_resnet101.pth',
    }
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 10,
    'brightness_range': 0.1,
    'contrast_range': 0.1,
    'saturation_range': 0.1,
}

# 性能配置
PERFORMANCE_CONFIG = {
    'batch_size': 1,           # 批处理大小
    'num_workers': 4,          # 数据加载线程数
    'pin_memory': True,        # 固定内存
    'mixed_precision': False,  # 混合精度
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'log_dir': './logs'
}

# 验证配置
VALIDATION_CONFIG = {
    'metrics': ['accuracy', 'iou', 'dice'],
    'save_confusion_matrix': True,
    'save_class_reports': True,
}

def get_model_config(dataset='pascal_voc', encoder='resnet101'):
    """获取特定数据集和编码器的模型配置"""
    config = MODEL_CONFIG.copy()
    
    if dataset == 'pascal_voc':
        config['num_classes'] = 21
        config['class_names'] = PASCAL_VOC_CLASSES['names']
        config['class_colors'] = PASCAL_VOC_CLASSES['colors']
    elif dataset == 'cityscapes':
        config['num_classes'] = 19
        # 可以添加Cityscapes的类别配置
    
    config['encoder_name'] = encoder
    return config

def get_output_paths(base_dir, create_dirs=True):
    """获取输出路径配置"""
    paths = {
        'base': base_dir,
        'masks': os.path.join(base_dir, 'masks'),
        'colored_masks': os.path.join(base_dir, 'colored_masks'),
        'overlays': os.path.join(base_dir, 'overlays'),
        'visualizations': os.path.join(base_dir, 'visualizations'),
        'logs': os.path.join(base_dir, 'logs'),
        'metrics': os.path.join(base_dir, 'metrics')
    }
    
    if create_dirs:
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
    
    return paths

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 验证模型配置
    if MODEL_CONFIG['encoder_name'] not in SUPPORTED_ENCODERS:
        errors.append(f"不支持的编码器: {MODEL_CONFIG['encoder_name']}")
    
    # 验证类别数量
    if len(PASCAL_VOC_CLASSES['names']) != len(PASCAL_VOC_CLASSES['colors']):
        errors.append("类别名称和颜色数量不匹配")
    
    # 验证图像尺寸
    if MODEL_CONFIG['input_size'][0] < IMAGE_CONFIG['min_size'][0] or \
       MODEL_CONFIG['input_size'][1] < IMAGE_CONFIG['min_size'][1]:
        errors.append("输入尺寸小于最小尺寸限制")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    return True

# 在导入时验证配置
if __name__ == '__main__':
    try:
        validate_config()
        print("配置验证通过")
    except ValueError as e:
        print(f"配置验证失败: {e}")