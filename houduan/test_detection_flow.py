#!/usr/bin/env python3
"""
检测流程测试脚本
用于诊断每个环节是否正常工作
"""

import os
import sys
import time
import subprocess
import tempfile
import io
from PIL import Image
import cv2
import numpy as np

# 添加 YOLOv5 路径
YOLOV5_PATH = '/Users/liangbaikai/Desktop/工作/huozai/yolov5/yolov5'
sys.path.insert(0, YOLOV5_PATH)

from stream_manager import StreamManager, StreamConfig
from detection_engine import DetectionEngine

def test_step_1_get_frame():
    """测试步骤 1：获取帧"""
    print("\n" + "="*60)
    print("步骤 1️⃣ ：获取摄像机帧数据")
    print("="*60)
    
    try:
        # 初始化流管理器
        stream_manager = StreamManager()
        
        # 添加流
        config = StreamConfig(
            name='1栋(5-10层)',
            rtsp_url='rtsp://admin:admin@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0',
            channel=1,
            subtype=0,
            username='admin',
            password='admin'
        )
        
        if not stream_manager.add_stream(config):
            print("❌ 添加流失败")
            return None
        
        # 启动流
        if not stream_manager.start_stream('1栋(5-10层)'):
            print("❌ 启动流失败")
            return None
        
        print("⏳ 等待 RTSP 连接建立...")
        time.sleep(3)
        
        # 获取帧
        frame_data = stream_manager.get_frame('1栋(5-10层)')
        
        if not frame_data:
            print("❌ 无法获取帧数据")
            return None
        
        print(f"✅ 成功获取帧数据")
        print(f"   - 大小：{len(frame_data)} 字节")
        print(f"   - 前 16 字节（十六进制）：{frame_data[:16].hex()}")
        
        return frame_data
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_2_ffmpeg_decode(frame_data):
    """测试步骤 2：ffmpeg 解码"""
    print("\n" + "="*60)
    print("步骤 2️⃣ ：使用 ffmpeg 解码 H.265")
    print("="*60)
    
    if not frame_data:
        print("❌ 没有帧数据")
        return None
    
    try:
        # 保存帧数据到临时文件
        with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as tmp:
            tmp.write(frame_data)
            tmp_path = tmp.name
        
        print(f"📝 帧数据已保存到：{tmp_path}")
        
        try:
            # 运行 ffmpeg
            process = subprocess.Popen(
                ['ffmpeg', '-loglevel', 'error',
                 '-f', 'hevc',
                 '-i', tmp_path,
                 '-vframes', '1',
                 '-f', 'image2',
                 '-c:v', 'mjpeg',
                 '-q:v', '5',
                 '-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            jpeg_data, stderr_data = process.communicate(timeout=5)
            
            if stderr_data:
                stderr_msg = stderr_data.decode('utf-8', errors='ignore')
                print(f"⚠️  ffmpeg stderr：{stderr_msg[:200]}")
            
            if not jpeg_data or len(jpeg_data) < 100:
                print(f"❌ ffmpeg 解码失败")
                print(f"   - JPEG 大小：{len(jpeg_data) if jpeg_data else 0} 字节")
                return None
            
            print(f"✅ ffmpeg 解码成功")
            print(f"   - JPEG 大小：{len(jpeg_data)} 字节")
            print(f"   - JPEG 前 4 字节（十六进制）：{jpeg_data[:4].hex()}")
            
            return jpeg_data
        
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_3_pil_load(jpeg_data):
    """测试步骤 3：PIL 加载 JPEG"""
    print("\n" + "="*60)
    print("步骤 3️⃣ ：PIL 加载 JPEG")
    print("="*60)
    
    if not jpeg_data:
        print("❌ 没有 JPEG 数据")
        return None
    
    try:
        img = Image.open(io.BytesIO(jpeg_data))
        
        print(f"✅ PIL 加载成功")
        print(f"   - 图像尺寸：{img.size}")
        print(f"   - 图像模式：{img.mode}")
        print(f"   - 图像格式：{img.format}")
        
        return img
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_4_opencv_convert(img):
    """测试步骤 4：OpenCV 转换"""
    print("\n" + "="*60)
    print("步骤 4️⃣ ：OpenCV 转换为 BGR")
    print("="*60)
    
    if img is None:
        print("❌ 没有图像")
        return None
    
    try:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        print(f"✅ OpenCV 转换成功")
        print(f"   - 帧形状：{frame.shape}")
        print(f"   - 数据类型：{frame.dtype}")
        print(f"   - 像素值范围：[{frame.min()}, {frame.max()}]")
        
        # 保存帧到文件用于调试
        debug_path = '/tmp/debug_frame.jpg'
        cv2.imwrite(debug_path, frame)
        print(f"   - 调试帧已保存到：{debug_path}")
        
        return frame
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_5_yolov5_detect(frame):
    """测试步骤 5：YOLOv5 检测"""
    print("\n" + "="*60)
    print("步骤 5️⃣ ：YOLOv5 检测")
    print("="*60)
    
    if frame is None:
        print("❌ 没有帧数据")
        return None
    
    try:
        # 初始化检测引擎
        model_path = '/Users/liangbaikai/Desktop/工作/huozai/yolov5/yolov5/best.pt'
        
        print(f"📝 加载模型：{model_path}")
        detection_engine = DetectionEngine(
            model_path=model_path,
            device='cpu',
            conf_threshold=0.4,
            iou_threshold=0.5
        )
        
        if detection_engine.model is None:
            print("❌ 模型加载失败")
            return None
        
        print("⏳ 执行检测...")
        start_time = time.time()
        result = detection_engine.detect(frame)
        elapsed = time.time() - start_time
        
        print(f"✅ YOLOv5 检测完成")
        print(f"   - 推理时间：{elapsed:.3f} 秒")
        print(f"   - 检测到 {len(result.detections)} 个目标")
        
        if result.detections:
            print("\n   检测结果：")
            for i, det in enumerate(result.detections, 1):
                print(f"   {i}. {det.class_name}")
                print(f"      - 置信度：{det.confidence:.4f}")
                print(f"      - 边界框：{det.bbox}")
        else:
            print("   ⚠️  没有检测到任何目标")
        
        return result
    
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🔍 检测流程完整测试")
    print("="*60)
    
    # 步骤 1：获取帧
    frame_data = test_step_1_get_frame()
    if not frame_data:
        print("\n❌ 测试中止：无法获取帧数据")
        return
    
    # 步骤 2：ffmpeg 解码
    jpeg_data = test_step_2_ffmpeg_decode(frame_data)
    if not jpeg_data:
        print("\n❌ 测试中止：ffmpeg 解码失败")
        return
    
    # 步骤 3：PIL 加载
    img = test_step_3_pil_load(jpeg_data)
    if img is None:
        print("\n❌ 测试中止：PIL 加载失败")
        return
    
    # 步骤 4：OpenCV 转换
    frame = test_step_4_opencv_convert(img)
    if frame is None:
        print("\n❌ 测试中止：OpenCV 转换失败")
        return
    
    # 步骤 5：YOLOv5 检测
    result = test_step_5_yolov5_detect(frame)
    if result is None:
        print("\n❌ 测试中止：YOLOv5 检测失败")
        return
    
    # 总结
    print("\n" + "="*60)
    print("✅ 测试完成")
    print("="*60)
    print(f"\n总结：")
    print(f"- 帧数据大小：{len(frame_data)} 字节")
    print(f"- JPEG 大小：{len(jpeg_data)} 字节")
    print(f"- 图像尺寸：{img.size}")
    print(f"- 帧形状：{frame.shape}")
    print(f"- 检测结果：{len(result.detections)} 个目标")
    
    if result.detections:
        print(f"\n✅ 检测成功！检测到火焰/烟雾")
    else:
        print(f"\n⚠️  没有检测到任何目标")
        print(f"   可能原因：")
        print(f"   1. 摄像机画面中没有火焰/烟雾")
        print(f"   2. 模型置信度阈值过高（当前：0.4）")
        print(f"   3. 模型权重不适合当前场景")


if __name__ == '__main__':
    main()
