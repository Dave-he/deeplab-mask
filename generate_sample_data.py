#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例数据生成脚本 - 创建测试图像用于验证DeepLab v3+功能

使用方法:
python generate_sample_data.py --output_dir ./data/input --num_images 10
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import math

def create_geometric_image(size=(512, 512), complexity='medium'):
    """创建包含几何形状的测试图像"""
    width, height = size
    
    # 创建基础图像
    image = Image.new('RGB', size, color=(135, 206, 235))  # 天蓝色背景
    draw = ImageDraw.Draw(image)
    
    if complexity == 'simple':
        num_shapes = random.randint(2, 4)
    elif complexity == 'medium':
        num_shapes = random.randint(4, 8)
    else:  # complex
        num_shapes = random.randint(8, 15)
    
    # 定义颜色列表
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
        (255, 165, 0),  # 橙色
        (128, 0, 128),  # 紫色
        (255, 192, 203), # 粉色
        (165, 42, 42),  # 棕色
    ]
    
    # 绘制随机形状
    for _ in range(num_shapes):
        color = random.choice(colors)
        shape_type = random.choice(['rectangle', 'circle', 'triangle', 'line'])
        
        if shape_type == 'rectangle':
            x1 = random.randint(0, width//2)
            y1 = random.randint(0, height//2)
            x2 = random.randint(x1 + 20, width)
            y2 = random.randint(y1 + 20, height)
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
        elif shape_type == 'circle':
            center_x = random.randint(50, width - 50)
            center_y = random.randint(50, height - 50)
            radius = random.randint(20, 80)
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], fill=color)
            
        elif shape_type == 'triangle':
            points = []
            for _ in range(3):
                x = random.randint(0, width)
                y = random.randint(0, height)
                points.append((x, y))
            draw.polygon(points, fill=color)
            
        elif shape_type == 'line':
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            width_line = random.randint(3, 10)
            draw.line([x1, y1, x2, y2], fill=color, width=width_line)
    
    return image

def create_nature_like_image(size=(512, 512)):
    """创建类似自然场景的图像"""
    width, height = size
    image = Image.new('RGB', size)
    draw = ImageDraw.Draw(image)
    
    # 天空渐变
    for y in range(height//3):
        color_intensity = int(255 * (1 - y / (height//3)))
        sky_color = (135 + color_intensity//4, 206, 235 + color_intensity//8)
        draw.line([(0, y), (width, y)], fill=sky_color)
    
    # 地面
    ground_color = (34, 139, 34)  # 森林绿
    draw.rectangle([0, height//3, width, height], fill=ground_color)
    
    # 添加"建筑物"
    for _ in range(random.randint(2, 5)):
        building_width = random.randint(40, 100)
        building_height = random.randint(60, 150)
        x = random.randint(0, width - building_width)
        y = height//3 - building_height
        
        building_color = random.choice([
            (128, 128, 128),  # 灰色
            (139, 69, 19),    # 棕色
            (255, 255, 255),  # 白色
            (220, 220, 220),  # 浅灰
        ])
        
        draw.rectangle([x, y, x + building_width, height//3], fill=building_color)
        
        # 添加窗户
        for window_y in range(y + 10, height//3 - 10, 20):
            for window_x in range(x + 10, x + building_width - 10, 15):
                if random.random() > 0.3:  # 70%概率有窗户
                    window_color = (255, 255, 0) if random.random() > 0.5 else (0, 0, 0)
                    draw.rectangle([window_x, window_y, window_x + 8, window_y + 12], 
                                 fill=window_color)
    
    # 添加"树木"
    for _ in range(random.randint(3, 8)):
        tree_x = random.randint(20, width - 20)
        tree_y = height//3
        
        # 树干
        trunk_color = (139, 69, 19)
        draw.rectangle([tree_x - 5, tree_y - 40, tree_x + 5, tree_y], fill=trunk_color)
        
        # 树冠
        crown_color = (0, 100, 0)
        crown_radius = random.randint(15, 30)
        draw.ellipse([tree_x - crown_radius, tree_y - 70, 
                     tree_x + crown_radius, tree_y - 10], fill=crown_color)
    
    # 添加"道路"
    road_color = (64, 64, 64)
    road_y = height//3 + random.randint(20, 50)
    draw.rectangle([0, road_y, width, road_y + 30], fill=road_color)
    
    # 道路标线
    line_color = (255, 255, 255)
    for x in range(0, width, 40):
        draw.rectangle([x, road_y + 12, x + 20, road_y + 18], fill=line_color)
    
    return image

def create_object_scene(size=(512, 512)):
    """创建包含常见物体的场景"""
    width, height = size
    image = Image.new('RGB', size, color=(240, 240, 240))  # 浅灰背景
    draw = ImageDraw.Draw(image)
    
    # 模拟桌面
    table_color = (139, 69, 19)
    draw.rectangle([0, height*2//3, width, height], fill=table_color)
    
    # 添加各种"物体"
    objects = [
        {'type': 'bottle', 'color': (0, 128, 0), 'size': (20, 60)},
        {'type': 'cup', 'color': (255, 255, 255), 'size': (30, 40)},
        {'type': 'book', 'color': (255, 0, 0), 'size': (40, 60)},
        {'type': 'phone', 'color': (0, 0, 0), 'size': (25, 50)},
        {'type': 'apple', 'color': (255, 0, 0), 'size': (25, 25)},
    ]
    
    for obj in objects:
        if random.random() > 0.3:  # 70%概率出现
            x = random.randint(50, width - 100)
            y = height*2//3 - obj['size'][1]
            w, h = obj['size']
            
            if obj['type'] == 'bottle':
                # 瓶子形状
                draw.rectangle([x, y, x + w, y + h], fill=obj['color'])
                draw.rectangle([x - 2, y - 5, x + w + 2, y + 5], fill=obj['color'])
            elif obj['type'] == 'cup':
                # 杯子形状
                draw.ellipse([x, y + h - 15, x + w, y + h], fill=obj['color'])
                draw.rectangle([x + 2, y, x + w - 2, y + h - 5], fill=obj['color'])
            else:
                # 其他物体用矩形表示
                draw.rectangle([x, y, x + w, y + h], fill=obj['color'])
    
    # 添加"人物"轮廓
    if random.random() > 0.5:
        person_x = random.randint(100, width - 150)
        person_y = height//4
        
        # 头部
        head_color = (255, 220, 177)
        draw.ellipse([person_x, person_y, person_x + 40, person_y + 40], fill=head_color)
        
        # 身体
        body_color = random.choice([(255, 0, 0), (0, 0, 255), (0, 128, 0)])
        draw.rectangle([person_x + 10, person_y + 40, person_x + 30, person_y + 100], 
                      fill=body_color)
        
        # 腿部
        leg_color = (0, 0, 139)
        draw.rectangle([person_x + 12, person_y + 100, person_x + 18, person_y + 140], 
                      fill=leg_color)
        draw.rectangle([person_x + 22, person_y + 100, person_x + 28, person_y + 140], 
                      fill=leg_color)
    
    return image

def add_noise_and_effects(image, noise_level=0.1):
    """添加噪声和效果"""
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 添加高斯噪声
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # 随机调整亮度
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        img_array = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def generate_sample_images(output_dir, num_images=10, image_size=(512, 512)):
    """生成示例图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_types = ['geometric', 'nature', 'objects']
    complexities = ['simple', 'medium', 'complex']
    
    print(f"正在生成 {num_images} 张示例图像...")
    
    for i in range(num_images):
        # 随机选择图像类型
        img_type = random.choice(image_types)
        
        if img_type == 'geometric':
            complexity = random.choice(complexities)
            image = create_geometric_image(image_size, complexity)
            filename = f"geometric_{complexity}_{i+1:03d}.png"
        elif img_type == 'nature':
            image = create_nature_like_image(image_size)
            filename = f"nature_scene_{i+1:03d}.png"
        else:  # objects
            image = create_object_scene(image_size)
            filename = f"object_scene_{i+1:03d}.png"
        
        # 添加随机效果
        if random.random() > 0.3:
            noise_level = random.uniform(0.05, 0.15)
            image = add_noise_and_effects(image, noise_level)
        
        # 保存图像
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"  生成: {filename}")
    
    print(f"\n✅ 成功生成 {num_images} 张示例图像")
    print(f"保存位置: {output_dir}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成DeepLab v3+测试用的示例图像')
    
    parser.add_argument('--output_dir', type=str, default='./data/input',
                       help='输出目录路径')
    parser.add_argument('--num_images', type=int, default=10,
                       help='生成图像数量')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                       help='图像尺寸 (width height)')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可重复生成）')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    # 生成示例图像
    generate_sample_images(
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=tuple(args.image_size)
    )
    
    print("\n📖 使用说明:")
    print(f"1. 测试单张图像: python demo.py --image {args.output_dir}/geometric_medium_001.png")
    print(f"2. 批量处理: python predict.py --input_dir {args.output_dir} --output_dir ./data/output")

if __name__ == '__main__':
    main()