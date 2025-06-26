#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLab v3+ 单张图像分割演示脚本

使用方法:
python demo.py --image /path/to/image.jpg --output /path/to/output/
"""

import os
import argparse
import matplotlib.pyplot as plt
from model import DeepLabV3Plus
from utils import load_image, save_image, create_overlay, visualize_prediction

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepLab v3+ 单张图像分割演示')
    
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--output', type=str, default='./demo_output',
                       help='输出目录路径')
    parser.add_argument('--weights', type=str, default=None,
                       help='模型权重文件路径（可选）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='使用的设备')
    parser.add_argument('--show_plot', action='store_true',
                       help='显示可视化结果')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 检查输入图像
    if not os.path.exists(args.image):
        print(f"错误: 输入图像不存在: {args.image}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 初始化模型
    print("正在初始化DeepLab v3+模型...")
    try:
        model = DeepLabV3Plus(
            weights=args.weights,
            device=args.device
        )
        print(f"模型初始化成功，使用设备: {model.device}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    # 加载图像
    print(f"正在加载图像: {args.image}")
    original_image = load_image(args.image)
    if original_image is None:
        print("图像加载失败")
        return
    
    print(f"图像尺寸: {original_image.shape}")
    
    # 进行预测
    print("正在进行分割预测...")
    try:
        prediction = model.predict(original_image)
        confidence = None
        
        # 也可以获取置信度
        # prediction, confidence = model.predict_with_confidence(original_image)
        
        print(f"预测完成，预测结果尺寸: {prediction.shape}")
        print(f"检测到的类别: {sorted(list(set(prediction.flatten())))}")
        
    except Exception as e:
        print(f"预测失败: {e}")
        return
    
    # 获取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # 保存单独的类别mask
    print("正在保存类别mask...")
    class_masks = model.get_class_masks(prediction)
    masks_dir = os.path.join(args.output, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    saved_masks = []
    for class_name, mask in class_masks.items():
        mask_path = os.path.join(masks_dir, f"{base_name}_{class_name}.png")
        if save_image(mask, mask_path):
            saved_masks.append((class_name, mask_path))
            print(f"  保存 {class_name} mask: {mask_path}")
    
    # 保存彩色mask
    print("正在保存彩色mask...")
    colored_mask = model.get_colored_mask(prediction)
    colored_mask_path = os.path.join(args.output, f"{base_name}_colored_mask.png")
    if save_image(colored_mask, colored_mask_path):
        print(f"  保存彩色mask: {colored_mask_path}")
    
    # 保存叠加图像
    print("正在保存叠加图像...")
    overlay = create_overlay(original_image, colored_mask, alpha=0.5)
    if overlay is not None:
        overlay_path = os.path.join(args.output, f"{base_name}_overlay.png")
        if save_image(overlay, overlay_path):
            print(f"  保存叠加图像: {overlay_path}")
    
    # 保存可视化结果
    print("正在生成可视化结果...")
    viz_path = os.path.join(args.output, f"{base_name}_visualization.png")
    visualize_prediction(original_image, prediction, model.class_names, viz_path)
    print(f"  保存可视化结果: {viz_path}")
    
    # 显示结果统计
    print("\n分割结果统计:")
    unique_classes, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = model.class_names[class_id] if class_id < len(model.class_names) else f"class_{class_id}"
        print(f"  {class_name}: {count} 像素 ({percentage:.2f}%)")
    
    # 显示可视化（如果指定）
    if args.show_plot:
        print("\n显示可视化结果...")
        visualize_prediction(original_image, prediction, model.class_names)
    
    print(f"\n演示完成！结果已保存到: {args.output}")

if __name__ == '__main__':
    import numpy as np
    main()