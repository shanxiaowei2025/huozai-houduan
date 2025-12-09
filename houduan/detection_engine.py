"""
YOLOv5 检测引擎
支持多摄像机并发检测
"""
import threading
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os

# YOLOv5 路径
YOLOV5_PATH = '/Users/liangbaikai/Desktop/工作/huozai/yolov5/yolov5'
MODEL_PATH = '/Users/liangbaikai/Desktop/工作/huozai/yolov5/yolov5/best.pt'

# 导入 YOLOv5
sys.path.insert(0, YOLOV5_PATH)

try:
    import torch
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
    TORCH_AVAILABLE = True
    print("[Detection] PyTorch 和 YOLOv5 导入成功")
except Exception as e:
    print(f"[Detection] 导入失败: {e}")
    TORCH_AVAILABLE = False


@dataclass
class Detection:
    """单个检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    
    def to_dict(self):
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox)
        }


@dataclass
class DetectionResult:
    """检测结果"""
    stream_name: str
    timestamp: float
    frame_id: int
    detections: List[Detection] = field(default_factory=list)
    inference_time: float = 0.0
    
    def to_dict(self):
        return {
            'stream_name': self.stream_name,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'detections': [d.to_dict() for d in self.detections],
            'inference_time': self.inference_time,
            'detection_count': len(self.detections)
        }


class DetectionEngine:
    """YOLOv5 检测引擎"""
    
    def __init__(self, model_path: str, device: str = 'cpu',
                 conf_threshold: float = 0.4, iou_threshold: float = 0.5,
                 img_size: int = 640):
        """
        初始化检测引擎
        
        Args:
            model_path: 模型权重路径
            device: 计算设备 ('cpu' 或 'cuda:0')
            conf_threshold: 置信度阈值
            iou_threshold: IOU 阈值
            img_size: 推理图像大小
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.class_names = ['fire', 'smoke']
        self.model = None
        self.device = None
        
        if not TORCH_AVAILABLE:
            print("[Detection] ❌ PyTorch 不可用，检测功能禁用")
            return
        
        try:
            print(f"[Detection] 加载模型: {model_path}")
            self.device = select_device(device)
            
            # 加载模型
            self.model = attempt_load(model_path, map_location=self.device)
            self.model.eval()
            
            # 修复 Upsample 问题
            self._fix_upsample()
            
            # 获取类名
            self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            print(f"[Detection] 模型类别: {self.class_names}")
            
            # 预热
            self._warmup()
            print("[Detection] ✅ 检测引擎初始化成功")
        except Exception as e:
            print(f"[Detection] ❌ 初始化失败: {e}")
            self.model = None
    
    def _fix_upsample(self):
        """修复 Upsample 层的 recompute_scale_factor 问题"""
        try:
            for m in self.model.modules():
                if m.__class__.__name__ == 'Upsample':
                    m.recompute_scale_factor = None
        except Exception as e:
            print(f"[Detection] 修复 Upsample 失败: {e}")
    
    def _warmup(self):
        """预热模型"""
        try:
            print("[Detection] 预热模型...")
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
            with torch.no_grad():
                self.model(dummy)
            print("[Detection] 模型预热完成")
        except Exception as e:
            print(f"[Detection] 预热失败（可忽略）: {e}")
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        检测单帧
        
        Args:
            frame: 输入帧 (BGR 格式)
            
        Returns:
            DetectionResult 对象
        """
        start_time = time.time()
        detections = []
        
        if self.model is None:
            return DetectionResult(
                stream_name='',
                timestamp=time.time(),
                frame_id=0,
                detections=[],
                inference_time=0
            )
        
        try:
            # 预处理
            img = self._preprocess(frame)
            
            # 推理
            with torch.no_grad():
                pred = self.model(img)[0]
            
            # NMS
            pred = non_max_suppression(
                pred,
                self.conf_threshold,
                self.iou_threshold,
                classes=None,
                agnostic=False
            )
            
            # 后处理
            detections = self._postprocess(pred, frame, img)
            
        except Exception as e:
            print(f"[Detection] 检测出错: {e}")
            detections = []
        
        inference_time = time.time() - start_time
        
        return DetectionResult(
            stream_name='',
            timestamp=time.time(),
            frame_id=0,
            detections=detections,
            inference_time=inference_time
        )
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整大小
        img = cv2.resize(frame, (self.img_size, self.img_size))
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为张量
        img = torch.from_numpy(img).to(self.device).float()
        img = img.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        img /= 255.0
        
        return img
    
    def _postprocess(self, pred, frame: np.ndarray, img: torch.Tensor) -> List[Detection]:
        """后处理检测结果"""
        detections = []
        
        if pred[0] is None or len(pred[0]) == 0:
            return detections
        
        h, w = frame.shape[:2]
        
        for det in pred[0]:
            # 缩放坐标
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (h, w))
            
            # 提取信息
            x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
            
            detection = Detection(
                class_id=int(cls_id),
                class_name=self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else f'class_{int(cls_id)}',
                confidence=float(conf),
                bbox=(float(x1), float(y1), float(x2), float(y2))
            )
            detections.append(detection)
        
        return detections


class MultiCameraDetector:
    """多摄像机检测器"""
    
    def __init__(self, engine: DetectionEngine, max_results: int = 100):
        """
        初始化多摄像机检测器
        
        Args:
            engine: 检测引擎
            max_results: 每个摄像机保留的最大结果数
        """
        self.engine = engine
        self.max_results = max_results
        
        # 每个摄像机的检测结果缓冲
        self.results: Dict[str, deque] = {}
        self.lock = threading.Lock()
        
        print("[Detector] 多摄像机检测器初始化完成")
    
    def register_stream(self, stream_name: str):
        """注册流"""
        with self.lock:
            if stream_name not in self.results:
                self.results[stream_name] = deque(maxlen=self.max_results)
                print(f"[Detector] 注册流: {stream_name}")
    
    def unregister_stream(self, stream_name: str):
        """注销流"""
        with self.lock:
            if stream_name in self.results:
                del self.results[stream_name]
                print(f"[Detector] 注销流: {stream_name}")
    
    def add_detection_result(self, stream_name: str, result: DetectionResult):
        """添加检测结果"""
        with self.lock:
            if stream_name in self.results:
                result.stream_name = stream_name
                self.results[stream_name].append(result)
    
    def get_latest_result(self, stream_name: str) -> Optional[DetectionResult]:
        """获取最新检测结果"""
        with self.lock:
            if stream_name in self.results and len(self.results[stream_name]) > 0:
                return self.results[stream_name][-1]
        return None
    
    def get_all_results(self, stream_name: str) -> List[DetectionResult]:
        """获取所有检测结果"""
        with self.lock:
            if stream_name in self.results:
                return list(self.results[stream_name])
        return []
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            stats = {}
            for stream_name, results in self.results.items():
                if len(results) > 0:
                    latest = results[-1]
                    stats[stream_name] = {
                        'total_results': len(results),
                        'latest_timestamp': latest.timestamp,
                        'latest_detection_count': len(latest.detections),
                        'latest_inference_time': latest.inference_time
                    }
            return stats
