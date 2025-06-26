import os
import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_image(image_path):
    """加载图像"""
    try:
        # 使用PIL加载图像
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None

def save_image(image, save_path):
    """保存图像"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            Image.fromarray(image).save(save_path)
        else:
            image.save(save_path)
        return True
    except Exception as e:
        print(f"保存图像失败 {save_path}: {e}")
        return False

def resize_image(image, target_size=(512, 512)):
    """调整图像大小"""
    if isinstance(image, np.ndarray):
        return cv2.resize(image, target_size)
    else:
        return image.resize(target_size)

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
    """获取目录下的所有图像文件"""
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)

def create_overlay(original_image, mask, alpha=0.5):
    """创建原图与mask的叠加图像"""
    if isinstance(original_image, np.ndarray) and isinstance(mask, np.ndarray):
        # 确保两个图像尺寸相同
        if original_image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        
        # 如果mask是灰度图，转换为RGB
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # 创建叠加图像
        overlay = cv2.addWeighted(original_image, 1-alpha, mask, alpha, 0)
        return overlay
    return None

def visualize_prediction(original_image, prediction, class_names, save_path=None):
    """可视化预测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示原图
    axes[0].imshow(original_image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示预测结果
    axes[1].imshow(prediction, cmap='tab20')
    axes[1].set_title('分割结果')
    axes[1].axis('off')
    
    # 添加类别图例
    unique_classes = np.unique(prediction)
    legend_elements = []
    for class_id in unique_classes:
        if class_id < len(class_names):
            legend_elements.append(plt.Rectangle((0,0),1,1, 
                                               facecolor=plt.cm.tab20(class_id/20), 
                                               label=class_names[class_id]))
    
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(prediction, ground_truth=None):
    """计算分割指标"""
    metrics = {}
    
    # 计算每个类别的像素数量
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        metrics[f'class_{class_id}_percentage'] = percentage
    
    # 如果有真实标签，计算更多指标
    if ground_truth is not None:
        # 计算准确率
        accuracy = np.mean(prediction == ground_truth)
        metrics['accuracy'] = accuracy
        
        # 计算每个类别的IoU
        for class_id in np.unique(np.concatenate([prediction.flatten(), ground_truth.flatten()])):
            pred_mask = prediction == class_id
            gt_mask = ground_truth == class_id
            
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            
            if union > 0:
                iou = intersection / union
                metrics[f'class_{class_id}_iou'] = iou
    
    return metrics

def batch_process_images(image_paths, model, output_dir, save_individual_masks=True, 
                        save_colored_masks=True, save_overlays=True, target_size=None):
    """批量处理图像"""
    results = []
    
    # 创建输出目录
    if save_individual_masks:
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
    
    if save_colored_masks:
        colored_masks_dir = os.path.join(output_dir, 'colored_masks')
        os.makedirs(colored_masks_dir, exist_ok=True)
    
    if save_overlays:
        overlays_dir = os.path.join(output_dir, 'overlays')
        os.makedirs(overlays_dir, exist_ok=True)
    
    # 处理每张图像
    for image_path in tqdm(image_paths, desc="处理图像"):
        try:
            # 加载图像
            original_image = load_image(image_path)
            if original_image is None:
                continue
            
            # 调整图像大小（如果指定）
            if target_size:
                processed_image = resize_image(original_image, target_size)
            else:
                processed_image = original_image
            
            # 预测
            prediction = model.predict(processed_image)
            
            # 如果调整了大小，将预测结果调整回原始大小
            if target_size and original_image.shape[:2] != processed_image.shape[:2]:
                prediction = cv2.resize(prediction.astype(np.uint8), 
                                      (original_image.shape[1], original_image.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 保存单独的类别mask
            if save_individual_masks:
                class_masks = model.get_class_masks(prediction)
                for class_name, mask in class_masks.items():
                    mask_path = os.path.join(masks_dir, f"{base_name}_{class_name}.png")
                    save_image(mask, mask_path)
            
            # 保存彩色mask
            if save_colored_masks:
                colored_mask = model.get_colored_mask(prediction)
                colored_mask_path = os.path.join(colored_masks_dir, f"{base_name}_colored.png")
                save_image(colored_mask, colored_mask_path)
            
            # 保存叠加图像
            if save_overlays:
                colored_mask = model.get_colored_mask(prediction)
                overlay = create_overlay(original_image, colored_mask)
                if overlay is not None:
                    overlay_path = os.path.join(overlays_dir, f"{base_name}_overlay.png")
                    save_image(overlay, overlay_path)
            
            # 计算指标
            metrics = calculate_metrics(prediction)
            
            results.append({
                'image_path': image_path,
                'prediction_shape': prediction.shape,
                'metrics': metrics
            })
            
        except Exception as e:
            print(f"处理图像失败 {image_path}: {e}")
            continue
    
    return results

def print_summary(results):
    """打印处理结果摘要"""
    if not results:
        print("没有成功处理的图像")
        return
    
    print(f"\n成功处理 {len(results)} 张图像")
    print("="*50)
    
    # 统计所有类别的平均占比
    all_metrics = {}
    for result in results:
        for key, value in result['metrics'].items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)
    
    print("\n平均类别占比:")
    for key, values in all_metrics.items():
        if 'percentage' in key:
            avg_value = np.mean(values)
            print(f"  {key}: {avg_value:.2f}%")