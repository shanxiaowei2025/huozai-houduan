"""
检测管理器 - 基于低延迟采集线程的检测
支持多摄像机并发检测，1-3 秒低延迟
"""
import threading
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, Optional, List
import sys
import os
from dataclasses import dataclass, field

# YOLOv5 路径
YOLOV5_PATH = '/Users/liangbaikai/Desktop/工作/huozai/yolov5/yolov5'
sys.path.insert(0, YOLOV5_PATH)

try:
    import torch
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[Detection] 导入失败: {e}")
    TORCH_AVAILABLE = False


@dataclass
class Detection:
    """单个检测结果"""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # x1, y1, x2, y2
    
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


class FrameBuffer:
    """线程安全的帧缓冲区 - 只保留最新帧"""
    
    def __init__(self, max_size=2):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def put(self, frame):
        """添加帧（自动丢弃旧帧）"""
        with self.lock:
            self.buffer.append(frame)
    
    def get(self):
        """获取最新帧"""
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer[-1].copy()
            return None
    
    def size(self):
        """获取缓冲区大小"""
        with self.lock:
            return len(self.buffer)


class StreamDetector:
    """单个流的检测器 - 低延迟版本"""
    
    def __init__(self, stream_name: str, rtsp_url: str, detection_engine, 
                 username: str = None, password: str = None):
        """初始化流检测器"""
        self.stream_name = stream_name
        self.rtsp_url = rtsp_url
        self.username = username
        self.password = password
        self.detection_engine = detection_engine
        
        # 构建完整 RTSP URL
        if username and password:
            self.full_rtsp_url = rtsp_url.replace(
                'rtsp://',
                f'rtsp://{username}:{password}@'
            )
        else:
            self.full_rtsp_url = rtsp_url
        
        # 帧缓冲区
        self.frame_buffer = FrameBuffer(max_size=2)
        
        # 采集线程
        self.capture_thread = None
        self.stop_capture = False
        self.cap = None
        
        # 检测结果
        self.latest_result = None
        self.frame_id = 0
        self.lock = threading.Lock()
        
        # 统计
        self.total_frames = 0
        self.total_detections = 0
        self.avg_inference_time = 0.0
    
    def start(self):
        """启动检测"""
        if self.capture_thread is not None:
            return
        
        self.stop_capture = False
        self.capture_thread = threading.Thread(target=self._capture_and_detect_loop, daemon=True)
        self.capture_thread.start()
        print(f"[Detection] 流 {self.stream_name} 检测已启动")
    
    def stop(self):
        """停止检测"""
        self.stop_capture = True
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        print(f"[Detection] 流 {self.stream_name} 检测已停止")
    
    def _capture_and_detect_loop(self):
        """采集和检测循环"""
        while not self.stop_capture:
            try:
                # 打开摄像机连接
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.full_rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"[Detection] 已连接到 {self.stream_name}")
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[Detection] {self.stream_name} 连接断开，重新连接...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)
                    continue
                
                # 添加到缓冲区
                self.frame_buffer.put(frame)
                
                # 检测
                self._detect_frame(frame)
                
            except Exception as e:
                print(f"[Detection] {self.stream_name} 错误: {e}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                time.sleep(1)
    
    def _detect_frame(self, frame):
        """检测单帧"""
        if self.detection_engine is None:
            return
        
        try:
            start_time = time.time()
            
            # 预处理
            img = cv2.resize(frame, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).to(self.detection_engine.device).float()
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor /= 255.0
            
            # 推理
            with torch.no_grad():
                pred = self.detection_engine.model(img_tensor)[0]
            
            # NMS
            pred = non_max_suppression(
                pred,
                self.detection_engine.conf_threshold,
                self.detection_engine.iou_threshold,
                classes=None,
                agnostic=False
            )
            
            # 后处理
            detections = []
            if pred[0] is not None and len(pred[0]) > 0:
                pred_array = pred[0].cpu().numpy()
                h, w = frame.shape[:2]
                
                for det in pred_array:
                    x1, y1, x2, y2, conf, cls_id = det
                    
                    # 缩放坐标
                    coords = scale_coords((640, 640), torch.tensor([[x1, y1, x2, y2]]), (h, w))
                    x1, y1, x2, y2 = coords[0].cpu().numpy()
                    
                    class_id = int(cls_id)
                    class_name = self.detection_engine.class_names[class_id] if class_id < len(self.detection_engine.class_names) else f'class_{class_id}'
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=(float(x1), float(y1), float(x2), float(y2))
                    ))
            
            inference_time = time.time() - start_time
            
            # 保存结果
            with self.lock:
                self.frame_id += 1
                self.latest_result = DetectionResult(
                    stream_name=self.stream_name,
                    timestamp=time.time(),
                    frame_id=self.frame_id,
                    detections=detections,
                    inference_time=inference_time
                )
                self.total_frames += 1
                self.total_detections += len(detections)
                
                # 更新平均推理时间
                if self.total_frames > 0:
                    self.avg_inference_time = (self.avg_inference_time * (self.total_frames - 1) + inference_time) / self.total_frames
        
        except Exception as e:
            print(f"[Detection] {self.stream_name} 检测错误: {e}")
    
    def get_latest_result(self) -> Optional[DetectionResult]:
        """获取最新检测结果"""
        with self.lock:
            return self.latest_result
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'stream_name': self.stream_name,
                'total_frames': self.total_frames,
                'total_detections': self.total_detections,
                'avg_inference_time': self.avg_inference_time
            }


class DetectionManager:
    """检测管理器 - 管理多个流的检测"""
    
    def __init__(self, model_path: str, device: str = 'cpu',
                 conf_threshold: float = 0.4, iou_threshold: float = 0.5):
        """初始化检测管理器"""
        self.model_path = model_path
        self.device_name = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 检测引擎
        self.detection_engine = None
        self._init_engine()
        
        # 流检测器
        self.detectors: Dict[str, StreamDetector] = {}
        self.lock = threading.Lock()
    
    def _init_engine(self):
        """初始化检测引擎"""
        if not TORCH_AVAILABLE:
            print("[Detection] PyTorch 不可用")
            return
        
        try:
            device = select_device(self.device_name)
            model = attempt_load(self.model_path, map_location=device)
            model.to(device).eval()
            
            # 修复 Upsample 层
            for m in model.modules():
                if m.__class__.__name__ == 'Upsample':
                    m.recompute_scale_factor = None
            
            # 获取类名
            class_names = model.module.names if hasattr(model, 'module') else model.names
            
            # 创建简单的引擎对象
            class Engine:
                pass
            
            engine = Engine()
            engine.model = model
            engine.device = device
            engine.conf_threshold = self.conf_threshold
            engine.iou_threshold = self.iou_threshold
            engine.class_names = class_names
            
            self.detection_engine = engine
            print(f"[Detection] 检测引擎初始化成功，类别: {class_names}")
            
        except Exception as e:
            print(f"[Detection] 检测引擎初始化失败: {e}")
    
    def add_stream(self, stream_name: str, rtsp_url: str, 
                   username: str = None, password: str = None) -> bool:
        """添加流检测"""
        if self.detection_engine is None:
            print("[Detection] 检测引擎未初始化")
            return False
        
        with self.lock:
            if stream_name in self.detectors:
                print(f"[Detection] 流 {stream_name} 已存在")
                return False
            
            detector = StreamDetector(
                stream_name=stream_name,
                rtsp_url=rtsp_url,
                detection_engine=self.detection_engine,
                username=username,
                password=password
            )
            detector.start()
            self.detectors[stream_name] = detector
            return True
    
    def remove_stream(self, stream_name: str) -> bool:
        """移除流检测"""
        with self.lock:
            if stream_name not in self.detectors:
                return False
            
            detector = self.detectors[stream_name]
            detector.stop()
            del self.detectors[stream_name]
            return True
    
    def get_detection_result(self, stream_name: str) -> Optional[DetectionResult]:
        """获取检测结果"""
        with self.lock:
            if stream_name not in self.detectors:
                return None
            
            return self.detectors[stream_name].get_latest_result()
    
    def get_all_results(self) -> List[DetectionResult]:
        """获取所有检测结果"""
        results = []
        with self.lock:
            for detector in self.detectors.values():
                result = detector.get_latest_result()
                if result:
                    results.append(result)
        return results
    
    def get_stats(self, stream_name: str = None):
        """获取统计信息"""
        with self.lock:
            if stream_name:
                if stream_name in self.detectors:
                    return self.detectors[stream_name].get_stats()
                return None
            else:
                return {
                    'total_streams': len(self.detectors),
                    'streams': [d.get_stats() for d in self.detectors.values()]
                }
    
    def shutdown(self):
        """关闭所有检测"""
        with self.lock:
            for detector in self.detectors.values():
                detector.stop()
            self.detectors.clear()
