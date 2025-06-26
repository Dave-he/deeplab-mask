# DeepLab v3+ 图像分割项目 Makefile

.PHONY: help install test demo generate-data clean setup run-demo run-batch

# 默认目标
help:
	@echo "DeepLab v3+ 图像分割项目"
	@echo "可用命令:"
	@echo "  make install        - 安装项目依赖"
	@echo "  make setup          - 完整项目设置（安装+生成测试数据）"
	@echo "  make test           - 运行模型测试"
	@echo "  make generate-data  - 生成示例测试数据"
	@echo "  make demo           - 运行单张图像演示"
	@echo "  make run-batch      - 运行批量处理演示"
	@echo "  make clean          - 清理输出文件"
	@echo "  make help           - 显示此帮助信息"

# 安装依赖
install:
	@echo "正在安装项目依赖..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ 依赖安装完成"

# 完整项目设置
setup: install
	@echo "正在进行项目设置..."
	python setup.py
	@echo "正在生成示例数据..."
	python generate_sample_data.py --output_dir ./data/input --num_images 5
	@echo "✅ 项目设置完成"

# 运行测试
test:
	@echo "正在运行模型测试..."
	python test_model.py

# 生成示例数据
generate-data:
	@echo "正在生成示例测试数据..."
	mkdir -p data/input
	python generate_sample_data.py --output_dir ./data/input --num_images 10
	@echo "✅ 示例数据生成完成"

# 运行单张图像演示
demo:
	@echo "正在运行单张图像演示..."
	@if [ ! -d "data/input" ] || [ -z "$$(ls -A data/input 2>/dev/null)" ]; then \
		echo "没有找到测试图像，正在生成..."; \
		make generate-data; \
	fi
	@FIRST_IMAGE=$$(ls data/input/*.png 2>/dev/null | head -1); \
	if [ -n "$$FIRST_IMAGE" ]; then \
		echo "使用图像: $$FIRST_IMAGE"; \
		python demo.py --image "$$FIRST_IMAGE" --output ./demo_output --show_plot; \
	else \
		echo "❌ 没有找到测试图像"; \
	fi

# 运行批量处理演示
run-batch:
	@echo "正在运行批量处理演示..."
	@if [ ! -d "data/input" ] || [ -z "$$(ls -A data/input 2>/dev/null)" ]; then \
		echo "没有找到测试图像，正在生成..."; \
		make generate-data; \
	fi
	mkdir -p data/output
	python predict.py --input_dir ./data/input --output_dir ./data/output
	@echo "✅ 批量处理完成，结果保存在 ./data/output"

# 快速测试（生成数据+运行演示）
quick-test: generate-data demo
	@echo "✅ 快速测试完成"

# 清理输出文件
clean:
	@echo "正在清理输出文件..."
	rm -rf data/output/*
	rm -rf demo_output/*
	rm -rf test_output/*
	rm -rf logs/*
	@echo "✅ 清理完成"

# 深度清理（包括生成的测试数据）
clean-all: clean
	@echo "正在进行深度清理..."
	rm -rf data/input/*
	rm -rf __pycache__/
	rm -rf *.pyc
	@echo "✅ 深度清理完成"

# 检查项目状态
status:
	@echo "项目状态检查:"
	@echo "Python版本: $$(python --version)"
	@echo "当前目录: $$(pwd)"
	@echo "输入图像数量: $$(ls data/input/*.png 2>/dev/null | wc -l | tr -d ' ')"
	@echo "输出结果: $$(if [ -d 'data/output' ]; then echo '存在'; else echo '不存在'; fi)"
	@python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 显示项目信息
info:
	@echo "DeepLab v3+ 图像分割项目"
	@echo "========================"
	@echo "功能: 使用DeepLab v3+模型进行图像语义分割"
	@echo "支持: 批量处理、单独类别mask生成、彩色mask、叠加可视化"
	@echo ""
	@echo "主要文件:"
	@echo "  model.py           - DeepLab v3+模型实现"
	@echo "  predict.py         - 批量预测脚本"
	@echo "  demo.py            - 单张图像演示"
	@echo "  utils.py           - 图像处理工具"
	@echo "  config.py          - 配置文件"
	@echo ""
	@echo "使用 'make help' 查看所有可用命令"

# 安装开发依赖
install-dev: install
	@echo "正在安装开发依赖..."
	pip install jupyter matplotlib seaborn
	@echo "✅ 开发依赖安装完成"

# 启动Jupyter notebook
notebook:
	@echo "正在启动Jupyter notebook..."
	jupyter notebook

# 创建项目目录结构
init-dirs:
	@echo "正在创建项目目录结构..."
	mkdir -p data/input data/output models logs demo_output test_output
	@echo "✅ 目录结构创建完成"