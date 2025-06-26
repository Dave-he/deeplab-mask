import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

class DeepLabV3Plus:
    """DeepLab v3+ 模型封装类"""
    
    def __init__(self, num_classes=21, encoder_name='resnet101', weights=None, device='auto'):
        """
        初始化DeepLab v3+模型
        
        Args:
            num_classes (int): 分割类别数，默认21（PASCAL VOC）
            encoder_name (str): 编码器名称，默认'resnet101'
            weights (str): 预训练权重路径
            device (str): 设备类型，'auto'自动选择
        """
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        
        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # 创建模型
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        
        # 加载自定义权重
        if weights:
            self.load_weights(weights)
            
        self.model.to(self.device)
        self.model.eval()
        
        # PASCAL VOC类别名称
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # 类别颜色映射
        self.colors = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ]
    
    def load_weights(self, weights_path):
        """加载模型权重"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载权重: {weights_path}")
        except Exception as e:
            print(f"加载权重失败: {e}")
            print("使用默认预训练权重")
    
    def preprocess(self, image):
        """图像预处理"""
        # 标准化参数（ImageNet）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # 转换为tensor并标准化
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.from_numpy(image).float()
            
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            
        # 确保通道顺序为RGB
        if tensor.shape[1] != 3:
            tensor = tensor.permute(0, 3, 1, 2)
            
        # 归一化到[0,1]
        if tensor.max() > 1:
            tensor = tensor / 255.0
            
        # 标准化
        tensor = (tensor - mean.to(tensor.device)) / std.to(tensor.device)
        
        return tensor.to(self.device)
    
    def predict(self, image):
        """预测单张图像"""
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess(image)
            
            # 预测
            output = self.model(input_tensor)
            
            # 获取预测结果
            pred = torch.argmax(output, dim=1)
            
            return pred.cpu().numpy()[0]
    
    def predict_with_confidence(self, image):
        """预测并返回置信度"""
        with torch.no_grad():
            # 预处理
            input_tensor = self.preprocess(image)
            
            # 预测
            output = self.model(input_tensor)
            
            # 应用softmax获取概率
            probs = F.softmax(output, dim=1)
            
            # 获取预测类别和置信度
            confidence, pred = torch.max(probs, dim=1)
            
            return pred.cpu().numpy()[0], confidence.cpu().numpy()[0]
    
    def get_class_masks(self, prediction):
        """获取每个类别的二值mask"""
        masks = {}
        for class_id in range(self.num_classes):
            if class_id in prediction:
                mask = (prediction == class_id).astype('uint8') * 255
                masks[self.class_names[class_id]] = mask
        return masks
    
    def get_colored_mask(self, prediction):
        """获取彩色mask"""
        h, w = prediction.shape
        colored_mask = torch.zeros((h, w, 3), dtype=torch.uint8)
        
        for class_id in range(self.num_classes):
            mask = prediction == class_id
            colored_mask[mask] = torch.tensor(self.colors[class_id], dtype=torch.uint8)
            
        return colored_mask.numpy()