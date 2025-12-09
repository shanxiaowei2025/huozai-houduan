"""
Flask API 服务器
提供 HTTP 接口用于管理 RTSP 媒体流
"""
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
import os
from dotenv import load_dotenv
from stream_manager import StreamManager, StreamConfig
from detection_manager import DetectionManager
import cv2
import numpy as np


# 加载环境变量
load_dotenv()

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 创建流管理器
stream_manager = StreamManager()

# ============================================================================
# 初始化检测管理器
# ============================================================================

MODEL_PATH = os.getenv('MODEL_WEIGHTS', '/Users/liangbaikai/Desktop/工作/huozai/yolov5/yolov5/best.pt')
DEVICE = os.getenv('DEVICE', 'cpu')
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.4))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.5))

detection_manager = None

try:
    detection_manager = DetectionManager(
        model_path=MODEL_PATH,
        device=DEVICE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    print("✅ 检测管理器初始化成功")
except Exception as e:
    print(f"❌ 检测管理器初始化失败: {e}")
    print(f"   模型路径: {MODEL_PATH}")
    print(f"   设备: {DEVICE}")

# 启动时自动加载配置文件中的流
def load_config_on_startup():
    """启动时加载配置文件"""
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            for stream_data in config.get('streams', []):
                if stream_data.get('enabled', True):
                    stream_config = StreamConfig(
                        name=stream_data['name'],
                        rtsp_url=stream_data['rtsp_url'],
                        channel=stream_data['channel'],
                        subtype=stream_data['subtype'],
                        username=stream_data.get('username'),
                        password=stream_data.get('password'),
                        enabled=stream_data.get('enabled', True)
                    )
                    if stream_manager.add_stream(stream_config):
                        print(f"✅ 自动加载流: {stream_config.name}")
                        # 自动启动流
                        stream_manager.start_stream(stream_config.name)
                        print(f"▶️  自动启动流: {stream_config.name}")
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")

# 在应用启动时加载配置
with app.app_context():
    load_config_on_startup()

# ============================================================================
# 配置路由
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'message': 'RTSP Media Stream Server is running'
    }), 200


# ============================================================================
# 流管理路由
# ============================================================================

@app.route('/api/streams', methods=['GET'])
def list_streams():
    """列出所有流"""
    streams = stream_manager.list_streams()
    statuses = stream_manager.get_all_status()
    
    return jsonify({
        'total': len(streams),
        'streams': [
            {
                'name': status.name,
                'connected': status.connected,
                'playing': status.playing,
                'packets_received': status.packets_received,
                'frames_received': status.frames_received,
                'bytes_received': status.bytes_received
            }
            for status in statuses
        ]
    }), 200


@app.route('/api/streams', methods=['POST'])
def add_stream():
    """添加新流"""
    try:
        data = request.get_json()
        
        # 验证必要字段
        required_fields = ['name', 'rtsp_url', 'channel', 'subtype']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields: {required_fields}'
            }), 400
        
        config = StreamConfig(
            name=data['name'],
            rtsp_url=data['rtsp_url'],
            channel=data['channel'],
            subtype=data['subtype'],
            username=data.get('username'),
            password=data.get('password'),
            enabled=data.get('enabled', True)
        )
        
        if stream_manager.add_stream(config):
            return jsonify({
                'message': f'Stream {config.name} added successfully',
                'stream': config.__dict__
            }), 201
        else:
            return jsonify({
                'error': f'Stream {config.name} already exists'
            }), 409
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/streams/<name>', methods=['GET'])
def get_stream_status(name):
    """获取流状态"""
    status = stream_manager.get_status(name)
    
    if not status:
        return jsonify({'error': f'Stream {name} not found'}), 404
    
    return jsonify({
        'name': status.name,
        'connected': status.connected,
        'playing': status.playing,
        'packets_received': status.packets_received,
        'frames_received': status.frames_received,
        'bytes_received': status.bytes_received,
        'last_frame_time': status.last_frame_time
    }), 200


@app.route('/api/streams/<name>', methods=['DELETE'])
def delete_stream(name):
    """删除流"""
    if stream_manager.remove_stream(name):
        return jsonify({
            'message': f'Stream {name} removed successfully'
        }), 200
    else:
        return jsonify({'error': f'Stream {name} not found'}), 404


@app.route('/api/streams/<name>/start', methods=['POST'])
def start_stream(name):
    """启动流"""
    if stream_manager.start_stream(name):
        return jsonify({
            'message': f'Stream {name} started'
        }), 200
    else:
        return jsonify({'error': f'Stream {name} not found'}), 404


@app.route('/api/streams/<name>/stop', methods=['POST'])
def stop_stream(name):
    """停止流"""
    if stream_manager.stop_stream(name):
        return jsonify({
            'message': f'Stream {name} stopped'
        }), 200
    else:
        return jsonify({'error': f'Stream {name} not found'}), 404


# ============================================================================
# 帧获取路由
# ============================================================================

@app.route('/api/streams/<name>/frame', methods=['GET'])
def get_frame(name):
    """获取流的最新帧（JPEG 格式）"""
    import subprocess
    import io
    from PIL import Image
    import tempfile
    import os
    
    frame_data = stream_manager.get_frame(name)
    
    if not frame_data:
        print(f"[Frame] 流 {name} 没有帧数据")
        # 返回黑色占位符
        img = Image.new('RGB', (640, 480), color=(0, 0, 0))
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return Response(img_io.getvalue(), mimetype='image/jpeg')
    
    print(f"[Frame] 获取到帧数据，大小: {len(frame_data)} 字节")
    
    try:
        # 创建临时文件来保存原始数据
        with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as tmp_file:
            tmp_file.write(frame_data)
            tmp_path = tmp_file.name
        
        try:
            # 使用 ffmpeg 将 H.265 转换为 JPEG
            # 关键参数：
            # -f hevc: 指定输入格式为原始 H.265
            # -vframes 1: 只提取第一帧
            # -q:v 5: 质量设置（1-31，越小越好）
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
            
            try:
                jpeg_data, stderr_data = process.communicate(timeout=5)
                
                if jpeg_data and len(jpeg_data) > 100:
                    print(f"[Frame] ffmpeg 转换成功，JPEG 大小: {len(jpeg_data)} 字节")
                    return Response(jpeg_data, mimetype='image/jpeg')
                else:
                    print(f"[Frame] ffmpeg 输出无效，大小: {len(jpeg_data) if jpeg_data else 0}")
                    if stderr_data:
                        stderr_msg = stderr_data.decode('utf-8', errors='ignore')
                        print(f"[Frame] ffmpeg stderr: {stderr_msg[:200]}")
                
            except subprocess.TimeoutExpired:
                print("[Frame] ffmpeg 超时")
                process.kill()
            except Exception as e:
                print(f"[Frame] ffmpeg 通信错误: {e}")
        
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # 如果 ffmpeg 失败，返回黑色占位符
        print("[Frame] 返回黑色占位符")
        img = Image.new('RGB', (640, 480), color=(0, 0, 0))
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return Response(img_io.getvalue(), mimetype='image/jpeg')
    
    except Exception as e:
        print(f"[Frame] 错误: {e}")
        img = Image.new('RGB', (640, 480), color=(0, 0, 0))
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        return Response(img_io.getvalue(), mimetype='image/jpeg')

@app.route('/api/streams/<name>/mjpeg', methods=['GET'])
def get_mjpeg_stream(name):
    """获取 MJPEG 流 - 低延迟直播"""
    def generate_mjpeg():
        """生成 MJPEG 流 - 使用 ffmpeg 直接转码"""
        import subprocess
        import time
        
        status = stream_manager.get_status(name)
        if not status or not status.connected:
            return
        
        # 获取 RTSP URL 和认证信息
        stream = None
        for s in stream_manager.streams.values():
            if s.config.name == name:
                stream = s
                break
        
        if not stream:
            return
        
        # 构建 ffmpeg 命令，直接从 RTSP 转码为 MJPEG
        rtsp_url = stream.config.rtsp_url
        if stream.config.username and stream.config.password:
            rtsp_url = rtsp_url.replace(
                'rtsp://',
                f'rtsp://{stream.config.username}:{stream.config.password}@'
            )
        
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-c:v', 'mjpeg',
            '-q:v', '5',
            '-f', 'mjpeg',
            '-'
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 读取 MJPEG 数据流
            while True:
                # 查找 JPEG 起始标记
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                
                # 简单的 JPEG 帧检测
                if b'\xff\xd8' in chunk:  # JPEG SOI
                    frame = chunk[chunk.find(b'\xff\xd8'):]
                    # 继续读取直到找到 EOI
                    while b'\xff\xd9' not in frame:
                        chunk = process.stdout.read(1024)
                        if not chunk:
                            break
                        frame += chunk
                    
                    if b'\xff\xd9' in frame:
                        frame = frame[:frame.find(b'\xff\xd9') + 2]
                        yield (b'--boundary\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' +
                               frame + b'\r\n')
        except Exception as e:
            print(f"[MJPEG] 错误: {e}")
        finally:
            try:
                process.terminate()
            except:
                pass
    
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=boundary')

@app.route('/api/streams/<name>/mjpeg_old', methods=['GET'])
def get_mjpeg_stream_old(name):
    """获取 MJPEG 流（备用方案）"""
    def generate_mjpeg():
        """生成 MJPEG 流"""
        try:
            import io
            from PIL import Image
            import subprocess
            import time
            
            last_frame_time = 0
            
            while True:
                status = stream_manager.get_status(name)
                if not status or not status.connected:
                    time.sleep(0.5)
                    continue
                
                # 每 100ms 获取一次帧
                if time.time() - last_frame_time < 0.1:
                    time.sleep(0.05)
                    continue
                
                frame_data = stream_manager.get_frame(name)
                if not frame_data:
                    time.sleep(0.05)
                    continue
                
                try:
                    # 使用 ffmpeg 将 H.264 转换为 JPEG
                    process = subprocess.Popen(
                        ['ffmpeg', '-i', 'pipe:0', '-f', 'image2', '-c:v', 'mjpeg', 'pipe:1'],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**8
                    )
                    
                    jpeg_data, _ = process.communicate(input=frame_data, timeout=1)
                    
                    if jpeg_data:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(jpeg_data)).encode() + b'\r\n\r\n' +
                               jpeg_data + b'\r\n')
                        last_frame_time = time.time()
                
                except Exception as e:
                    print(f"Error converting frame: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"MJPEG stream error: {e}")
    
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# 配置管理路由
# ============================================================================

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取配置"""
    config = {
        'server': {
            'host': os.getenv('HOST', '0.0.0.0'),
            'port': int(os.getenv('PORT', 5000)),
            'debug': os.getenv('DEBUG', 'False').lower() == 'true'
        },
        'rtsp': {
            'default_port': 554,
            'timeout': int(os.getenv('RTSP_TIMEOUT', 10))
        }
    }
    return jsonify(config), 200


@app.route('/api/config/streams', methods=['POST'])
def load_streams_config():
    """从配置文件加载流配置"""
    try:
        data = request.get_json()
        
        if 'streams' not in data:
            return jsonify({'error': 'Missing streams field'}), 400
        
        added = 0
        for stream_data in data['streams']:
            config = StreamConfig(
                name=stream_data['name'],
                rtsp_url=stream_data['rtsp_url'],
                channel=stream_data['channel'],
                subtype=stream_data['subtype'],
                username=stream_data.get('username'),
                password=stream_data.get('password'),
                enabled=stream_data.get('enabled', True)
            )
            
            if stream_manager.add_stream(config):
                added += 1
        
        return jsonify({
            'message': f'Added {added} streams',
            'total': len(data['streams'])
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============================================================================
# 检测路由
# ============================================================================

@app.route('/api/streams/<name>/detect', methods=['GET'])
def get_detection_result(name):
    """获取流的最新检测结果 - 实时检测当前帧"""
    if not detection_engine or not detector:
        return jsonify({'error': 'Detection engine not available'}), 503
    
    try:
        # ============ 步骤 1：获取当前帧 ============
        frame_data = stream_manager.get_frame(name)
        if not frame_data:
            print(f"[Detection] ❌ 步骤1 失败：无法获取帧数据")
            return jsonify({
                'stream_name': name,
                'detections': [],
                'message': 'No frame available'
            }), 200
        
        print(f"[Detection] ✅ 步骤1：获取帧数据成功，大小: {len(frame_data)} 字节")
        
        # ============ 步骤 2：使用 ffmpeg 解码 H.265 ============
        import subprocess
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as tmp:
            tmp.write(frame_data)
            tmp_path = tmp.name
        
        try:
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
                print(f"[Detection] ⚠️  ffmpeg stderr: {stderr_msg[:500]}")
            
            if not jpeg_data or len(jpeg_data) < 100:
                print(f"[Detection] ❌ 步骤2 失败：ffmpeg 解码失败，JPEG 大小: {len(jpeg_data) if jpeg_data else 0}")
                return jsonify({
                    'stream_name': name,
                    'detections': [],
                    'message': 'ffmpeg decode failed'
                }), 200
            
            print(f"[Detection] ✅ 步骤2：ffmpeg 解码成功，JPEG 大小: {len(jpeg_data)} 字节")
            
            # ============ 步骤 3：PIL 加载 JPEG ============
            import io
            from PIL import Image
            
            try:
                img = Image.open(io.BytesIO(jpeg_data))
                print(f"[Detection] ✅ 步骤3：PIL 加载成功，图像尺寸: {img.size}, 模式: {img.mode}")
            except Exception as e:
                print(f"[Detection] ❌ 步骤3 失败：PIL 加载失败 - {e}")
                return jsonify({
                    'stream_name': name,
                    'detections': [],
                    'message': f'PIL load failed: {str(e)}'
                }), 200
            
            # ============ 步骤 4：OpenCV 转换 ============
            try:
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                print(f"[Detection] ✅ 步骤4：OpenCV 转换成功，帧形状: {frame.shape}, dtype: {frame.dtype}")
            except Exception as e:
                print(f"[Detection] ❌ 步骤4 失败：OpenCV 转换失败 - {e}")
                return jsonify({
                    'stream_name': name,
                    'detections': [],
                    'message': f'OpenCV conversion failed: {str(e)}'
                }), 200
            
            # ============ 步骤 5：YOLOv5 检测 ============
            print(f"[Detection] 开始 YOLOv5 检测...")
            result = detection_engine.detect(frame)
            detector.add_detection_result(name, result)
            
            print(f"[Detection] ✅ 步骤5：YOLOv5 检测完成，检测到 {len(result.detections)} 个目标")
            if result.detections:
                for det in result.detections:
                    print(f"  - {det.class_name}: {det.confidence:.2f} @ {det.bbox}")
            
            return jsonify(result.to_dict()), 200
        
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    except Exception as e:
        print(f"[Detection] ❌ 检测出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'stream_name': name
        }), 500


@app.route('/api/streams/<name>/detect/batch', methods=['POST'])
def batch_detect(name):
    """批量检测帧"""
    if not detector:
        return jsonify({'error': 'Detection engine not available'}), 503
    
    try:
        # 获取最新帧
        frame_data = stream_manager.get_frame(name)
        if not frame_data:
            return jsonify({'error': 'No frame available'}), 404
        
        # 解码帧
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # 检测
        result = detection_engine.detect(frame)
        detector.add_detection_result(name, result)
        
        return jsonify(result.to_dict()), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============================================================================
# 错误处理
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# 检测路由
# ============================================================================

@app.route('/api/detection/start/<stream_name>', methods=['POST'])
def start_detection(stream_name):
    """启动流的检测"""
    if detection_manager is None:
        return jsonify({'error': '检测管理器未初始化'}), 500
    
    # 获取流配置
    stream = None
    for s in stream_manager.streams.values():
        if s.config.name == stream_name:
            stream = s
            break
    
    if not stream:
        return jsonify({'error': f'流 {stream_name} 不存在'}), 404
    
    # 添加流到检测管理器
    success = detection_manager.add_stream(
        stream_name=stream_name,
        rtsp_url=stream.config.rtsp_url,
        username=stream.config.username,
        password=stream.config.password
    )
    
    if success:
        return jsonify({
            'message': f'流 {stream_name} 检测已启动',
            'stream_name': stream_name
        }), 200
    else:
        return jsonify({'error': f'流 {stream_name} 检测启动失败'}), 400


@app.route('/api/detection/stop/<stream_name>', methods=['POST'])
def stop_detection(stream_name):
    """停止流的检测"""
    if detection_manager is None:
        return jsonify({'error': '检测管理器未初始化'}), 500
    
    success = detection_manager.remove_stream(stream_name)
    
    if success:
        return jsonify({
            'message': f'流 {stream_name} 检测已停止',
            'stream_name': stream_name
        }), 200
    else:
        return jsonify({'error': f'流 {stream_name} 不存在'}), 404


@app.route('/api/detection/<stream_name>', methods=['GET'])
def get_detection(stream_name):
    """获取流的最新检测结果"""
    if detection_manager is None:
        return jsonify({'error': '检测管理器未初始化'}), 500
    
    result = detection_manager.get_detection_result(stream_name)
    
    if result:
        return jsonify(result.to_dict()), 200
    else:
        return jsonify({
            'stream_name': stream_name,
            'detections': [],
            'detection_count': 0,
            'message': '暂无检测结果'
        }), 200


@app.route('/api/detection/all', methods=['GET'])
def get_all_detections():
    """获取所有流的检测结果"""
    if detection_manager is None:
        return jsonify({'error': '检测管理器未初始化'}), 500
    
    results = detection_manager.get_all_results()
    
    return jsonify({
        'total_streams': len(results),
        'results': [r.to_dict() for r in results]
    }), 200


@app.route('/api/detection/stats', methods=['GET'])
def get_detection_stats():
    """获取检测统计信息"""
    if detection_manager is None:
        return jsonify({'error': '检测管理器未初始化'}), 500
    
    stats = detection_manager.get_stats()
    
    return jsonify(stats), 200


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print(f"Starting RTSP Media Stream Server on {host}:{port}")
    print(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)
