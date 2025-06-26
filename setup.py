#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装脚本 - DeepLab v3+ 图像分割项目

使用方法:
python setup.py
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("请使用Python 3.7或更高版本")
        return False
    else:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True

def check_system_info():
    """检查系统信息"""
    print("\n检查系统信息...")
    system = platform.system()
    machine = platform.machine()
    print(f"操作系统: {system}")
    print(f"架构: {machine}")
    
    # 检查是否有CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查CUDA")

def install_requirements():
    """安装依赖包"""
    print("\n安装依赖包...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ 找不到 {requirements_file}")
        return False
    
    try:
        # 升级pip
        print("升级pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装依赖
        print("安装项目依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        print("✅ 依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def verify_installation():
    """验证安装"""
    print("\n验证安装...")
    
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy', 
        'cv2', 'matplotlib', 'tqdm', 'segmentation_models_pytorch'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ 以下包导入失败: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ 所有依赖包验证通过")
        return True

def create_directories():
    """创建必要的目录"""
    print("\n创建项目目录...")
    
    directories = [
        'data/input',
        'data/output',
        'models',
        'logs',
        'test_output'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
        except Exception as e:
            print(f"❌ 创建目录失败 {directory}: {e}")

def run_basic_test():
    """运行基本测试"""
    print("\n运行基本测试...")
    
    try:
        # 测试模型导入
        from model import DeepLabV3Plus
        print("✅ 模型导入成功")
        
        # 测试工具导入
        from utils import load_image, save_image
        print("✅ 工具导入成功")
        
        # 测试配置导入
        from config import get_model_config
        print("✅ 配置导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
        return False

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("🎉 安装完成！")
    print("="*60)
    
    print("\n📖 使用说明:")
    print("""
1. 测试模型功能:
   python test_model.py

2. 单张图像演示:
   python demo.py --image /path/to/image.jpg --output ./demo_output

3. 批量处理图像:
   python predict.py --input_dir /path/to/images --output_dir /path/to/output

4. 查看帮助:
   python predict.py --help
   python demo.py --help

📁 项目结构:
   ├── model.py              # DeepLab v3+ 模型实现
   ├── utils.py              # 图像处理工具
   ├── config.py             # 配置文件
   ├── predict.py            # 批量预测脚本
   ├── demo.py               # 单张图像演示
   ├── test_model.py         # 模型测试脚本
   ├── data/                 # 数据目录
   │   ├── input/           # 输入图像
   │   └── output/          # 输出结果
   └── models/              # 模型权重文件

💡 提示:
   - 首次运行会自动下载预训练权重
   - 支持CPU和GPU运行
   - 输出包括单独类别mask、彩色mask和叠加图像
   """)

def main():
    """主安装函数"""
    print("DeepLab v3+ 图像分割项目安装程序")
    print("="*60)
    
    # 检查Python版本
    if not check_python_version():
        return
    
    # 检查系统信息
    check_system_info()
    
    # 安装依赖
    if not install_requirements():
        print("\n❌ 安装失败，请检查错误信息")
        return
    
    # 验证安装
    if not verify_installation():
        print("\n❌ 验证失败，请检查依赖安装")
        return
    
    # 创建目录
    create_directories()
    
    # 运行基本测试
    if not run_basic_test():
        print("\n❌ 基本测试失败")
        return
    
    # 打印使用说明
    print_usage_instructions()

if __name__ == '__main__':
    main()