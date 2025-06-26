#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLab v3+ 批量图像分割预测脚本

使用方法:
python predict.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import argparse
import time
from model import DeepLabV3Plus
from utils import get_image_files, batch_process_images, print_summary

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepLab v3+ 批量图像分割')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入图像文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出结果文件夹路径')
    parser.add_argument('--weights', type=str, default=None,
                       help='模型权重文件路径（可选）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='使用的设备')
    parser.add_argument('--num_classes', type=int, default=21,
                       help='分割类别数')
    parser.add_argument('--encoder', type=str, default='resnet101',
                       choices=['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d'],
                       help='编码器类型')
    parser.add_argument('--target_size', type=int, nargs=2, default=None,
                       help='目标图像尺寸 (width height)，例如: --target_size 512 512')
    parser.add_argument('--save_individual_masks', action='store_true', default=True,
                       help='保存每个类别的单独mask')
    parser.add_argument('--save_colored_masks', action='store_true', default=True,
                       help='保存彩色mask')
    parser.add_argument('--save_overlays', action='store_true', default=True,
                       help='保存叠加图像')
    parser.add_argument('--extensions', type=str, nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='支持的图像文件扩展名')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图像文件列表
    print(f"正在扫描图像文件: {args.input_dir}")
    image_files = get_image_files(args.input_dir, tuple(args.extensions))
    
    if not image_files:
        print(f"错误: 在目录 {args.input_dir} 中没有找到支持的图像文件")
        print(f"支持的扩展名: {args.extensions}")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 初始化模型
    print("正在初始化DeepLab v3+模型...")
    try:
        model = DeepLabV3Plus(
            num_classes=args.num_classes,
            encoder_name=args.encoder,
            weights=args.weights,
            device=args.device
        )
        print(f"模型初始化成功，使用设备: {model.device}")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return
    
    # 处理目标尺寸
    target_size = None
    if args.target_size:
        target_size = tuple(args.target_size)
        print(f"将图像调整为: {target_size}")
    
    # 开始批量处理
    print("\n开始批量处理图像...")
    start_time = time.time()
    
    try:
        results = batch_process_images(
            image_paths=image_files,
            model=model,
            output_dir=args.output_dir,
            save_individual_masks=args.save_individual_masks,
            save_colored_masks=args.save_colored_masks,
            save_overlays=args.save_overlays,
            target_size=target_size
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 打印结果摘要
        print_summary(results)
        print(f"\n总处理时间: {processing_time:.2f} 秒")
        print(f"平均每张图像: {processing_time/len(image_files):.2f} 秒")
        print(f"\n结果已保存到: {args.output_dir}")
        
        # 打印输出目录结构
        print("\n输出目录结构:")
        if args.save_individual_masks:
            print(f"  - masks/: 每个类别的二值mask图")
        if args.save_colored_masks:
            print(f"  - colored_masks/: 彩色整体mask图")
        if args.save_overlays:
            print(f"  - overlays/: 原图与mask叠加的可视化结果")
            
    except Exception as e:
        print(f"批量处理过程中发生错误: {e}")
        return

if __name__ == '__main__':
    main()