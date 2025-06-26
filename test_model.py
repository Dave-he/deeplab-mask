#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试脚本 - 验证DeepLab v3+模型功能

使用方法:
python test_model.py
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model import DeepLabV3Plus
from utils import save_image, create_overlay
from config import get_model_config, get_output_paths

def create_test_image(size=(512, 512)):
    """创建测试图像"""
    # 创建一个简单的测试图像
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # 添加一些几何形状
    h, w = size
    
    # 背景渐变
    for i in range(h):
        for j in range(w):
            image[i, j] = [int(255 * i / h), int(255 * j / w), 128]
    
    # 添加矩形
    image[h//4:3*h//4, w//4:3*w//4] = [255, 0, 0]  # 红色矩形
    
    # 添加圆形
    center_x, center_y = w//2, h//2
    radius = min(w, h) // 8
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = [0, 255, 0]  # 绿色圆形
    
    return image

def test_model_initialization():
    """测试模型初始化"""
    print("测试1: 模型初始化")
    print("-" * 30)
    
    try:
        # 测试默认配置
        model = DeepLabV3Plus()
        print(f"✓ 默认配置初始化成功")
        print(f"  设备: {model.device}")
        print(f"  类别数: {model.num_classes}")
        print(f"  编码器: {model.encoder_name}")
        
        # 测试不同编码器
        encoders = ['resnet50', 'resnet101']
        for encoder in encoders:
            try:
                model = DeepLabV3Plus(encoder_name=encoder)
                print(f"✓ {encoder} 编码器初始化成功")
            except Exception as e:
                print(f"✗ {encoder} 编码器初始化失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False

def test_image_processing():
    """测试图像处理功能"""
    print("\n测试2: 图像处理")
    print("-" * 30)
    
    try:
        # 创建测试图像
        test_image = create_test_image()
        print(f"✓ 测试图像创建成功，尺寸: {test_image.shape}")
        
        # 初始化模型
        model = DeepLabV3Plus()
        
        # 测试预处理
        processed = model.preprocess(test_image)
        print(f"✓ 图像预处理成功，输出尺寸: {processed.shape}")
        print(f"  数据类型: {processed.dtype}")
        print(f"  数值范围: [{processed.min():.3f}, {processed.max():.3f}]")
        
        return True, test_image, model
        
    except Exception as e:
        print(f"✗ 图像处理失败: {e}")
        return False, None, None

def test_prediction(test_image, model):
    """测试预测功能"""
    print("\n测试3: 模型预测")
    print("-" * 30)
    
    try:
        # 测试基本预测
        prediction = model.predict(test_image)
        print(f"✓ 基本预测成功，输出尺寸: {prediction.shape}")
        print(f"  预测类别范围: [{prediction.min()}, {prediction.max()}]")
        print(f"  检测到的类别: {sorted(list(set(prediction.flatten())))}")
        
        # 测试带置信度的预测
        pred_conf, confidence = model.predict_with_confidence(test_image)
        print(f"✓ 置信度预测成功")
        print(f"  平均置信度: {confidence.mean():.3f}")
        print(f"  置信度范围: [{confidence.min():.3f}, {confidence.max():.3f}]")
        
        return True, prediction
        
    except Exception as e:
        print(f"✗ 模型预测失败: {e}")
        return False, None

def test_mask_generation(model, prediction):
    """测试mask生成功能"""
    print("\n测试4: Mask生成")
    print("-" * 30)
    
    try:
        # 测试类别mask生成
        class_masks = model.get_class_masks(prediction)
        print(f"✓ 类别mask生成成功，生成了 {len(class_masks)} 个类别mask")
        
        for class_name, mask in class_masks.items():
            unique_values = np.unique(mask)
            print(f"  {class_name}: 尺寸 {mask.shape}, 值域 {unique_values}")
        
        # 测试彩色mask生成
        colored_mask = model.get_colored_mask(prediction)
        print(f"✓ 彩色mask生成成功，尺寸: {colored_mask.shape}")
        print(f"  数据类型: {colored_mask.dtype}")
        
        return True, class_masks, colored_mask
        
    except Exception as e:
        print(f"✗ Mask生成失败: {e}")
        return False, None, None

def test_output_saving(test_image, prediction, class_masks, colored_mask):
    """测试输出保存功能"""
    print("\n测试5: 输出保存")
    print("-" * 30)
    
    try:
        # 创建测试输出目录
        output_dir = "./test_output"
        paths = get_output_paths(output_dir)
        
        # 保存测试图像
        test_image_path = os.path.join(paths['base'], 'test_image.png')
        save_image(test_image, test_image_path)
        print(f"✓ 测试图像保存成功: {test_image_path}")
        
        # 保存类别masks
        saved_masks = 0
        for class_name, mask in class_masks.items():
            mask_path = os.path.join(paths['masks'], f'test_{class_name}.png')
            if save_image(mask, mask_path):
                saved_masks += 1
        print(f"✓ 类别mask保存成功: {saved_masks}/{len(class_masks)} 个")
        
        # 保存彩色mask
        colored_mask_path = os.path.join(paths['colored_masks'], 'test_colored.png')
        save_image(colored_mask, colored_mask_path)
        print(f"✓ 彩色mask保存成功: {colored_mask_path}")
        
        # 保存叠加图像
        overlay = create_overlay(test_image, colored_mask)
        if overlay is not None:
            overlay_path = os.path.join(paths['overlays'], 'test_overlay.png')
            save_image(overlay, overlay_path)
            print(f"✓ 叠加图像保存成功: {overlay_path}")
        
        return True, output_dir
        
    except Exception as e:
        print(f"✗ 输出保存失败: {e}")
        return False, None

def test_performance():
    """测试性能"""
    print("\n测试6: 性能测试")
    print("-" * 30)
    
    try:
        import time
        
        # 创建测试数据
        test_images = [create_test_image() for _ in range(5)]
        model = DeepLabV3Plus()
        
        # 预热
        _ = model.predict(test_images[0])
        
        # 性能测试
        start_time = time.time()
        for img in test_images:
            _ = model.predict(img)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(test_images)
        
        print(f"✓ 性能测试完成")
        print(f"  总时间: {total_time:.3f} 秒")
        print(f"  平均每张: {avg_time:.3f} 秒")
        print(f"  FPS: {1/avg_time:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("DeepLab v3+ 模型测试")
    print("=" * 50)
    
    test_results = []
    
    # 测试1: 模型初始化
    result1 = test_model_initialization()
    test_results.append(("模型初始化", result1))
    
    if not result1:
        print("\n模型初始化失败，跳过后续测试")
        return
    
    # 测试2: 图像处理
    result2, test_image, model = test_image_processing()
    test_results.append(("图像处理", result2))
    
    if not result2:
        print("\n图像处理失败，跳过后续测试")
        return
    
    # 测试3: 模型预测
    result3, prediction = test_prediction(test_image, model)
    test_results.append(("模型预测", result3))
    
    if not result3:
        print("\n模型预测失败，跳过后续测试")
        return
    
    # 测试4: Mask生成
    result4, class_masks, colored_mask = test_mask_generation(model, prediction)
    test_results.append(("Mask生成", result4))
    
    if not result4:
        print("\nMask生成失败，跳过后续测试")
        return
    
    # 测试5: 输出保存
    result5, output_dir = test_output_saving(test_image, prediction, class_masks, colored_mask)
    test_results.append(("输出保存", result5))
    
    # 测试6: 性能测试
    result6 = test_performance()
    test_results.append(("性能测试", result6))
    
    # 打印测试总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(test_results)} 项测试通过")
    
    if result5 and output_dir:
        print(f"\n测试输出已保存到: {output_dir}")
    
    if passed == len(test_results):
        print("\n🎉 所有测试通过！模型工作正常。")
    else:
        print(f"\n⚠️  有 {len(test_results) - passed} 项测试失败，请检查相关功能。")

if __name__ == '__main__':
    main()