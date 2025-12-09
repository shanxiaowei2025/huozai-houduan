import cv2
import torch
import threading
from flask import Flask, render_template_string, Response
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, plot_one_box
from utils.torch_utils import select_device

app = Flask(__name__)

# 全局变量
frame_buffer = None
lock = threading.Lock()
model = None
device = None

def load_model(weights, device_name):
    global model, device
    device = select_device(device_name)
    model = attempt_load(weights, map_location=device)
    model.eval()
    return model

def detect_frame(frame, imgsz=640, conf_thres=0.4, iou_thres=0.5):
    """对单帧进行检测"""
    h, w = frame.shape[:2]
    
    # 调整大小
    img = cv2.resize(frame, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.copy()  # 修复负步长问题
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    # 绘制结果
    if pred is not None:
        for det in pred:
            if det is not None and len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)
    
    return frame

def capture_stream(rtsp_url):
    """从RTSP流捕获并检测"""
    global frame_buffer
    
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        print(f"正在连接到流... (尝试 {retry_count + 1}/{max_retries})")
        cap = cv2.VideoCapture(rtsp_url)
        
        # 设置超时
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"❌ 无法打开流: {rtsp_url}")
            retry_count += 1
            import time
            time.sleep(2)
            continue
        
        print(f"✅ 已连接到流: {rtsp_url}")
        retry_count = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取帧，重新连接...")
                cap.release()
                break
            
            frame_count += 1
            
            # 缩小帧以加快处理速度
            frame = cv2.resize(frame, (640, 480))
            
            # 检测
            try:
                detected_frame = detect_frame(frame, imgsz=640, conf_thres=0.4)
                
                # 保存到缓冲区
                with lock:
                    frame_buffer = detected_frame.copy()
                
                if frame_count % 30 == 0:
                    print(f"✅ 已处理 {frame_count} 帧")
            except Exception as e:
                print(f"❌ 检测出错: {e}")
                continue
        
        retry_count += 1
        import time
        time.sleep(2)

def generate_frames():
    """生成MJPEG流"""
    while True:
        with lock:
            if frame_buffer is None:
                continue
            frame = frame_buffer.copy()
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
               frame_bytes + b'\r\n')

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv5 火灾烟雾检测</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background: #1a1a1a;
            }
            .container {
                text-align: center;
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
            }
            h1 {
                color: #00ff00;
                margin: 0 0 20px 0;
            }
            img {
                max-width: 800px;
                width: 100%;
                border: 2px solid #00ff00;
                border-radius: 5px;
            }
            .status {
                color: #00ff00;
                margin-top: 20px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔥 YOLOv5 火灾烟雾实时检测</h1>
            <img src="/video_feed" alt="实时检测画面">
            <div class="status">
                <p>✅ 实时检测中...</p>
                <p>绿色框 = 检测到的火灾/烟雾</p>
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='192.168.1.108', help='摄像头IP地址')
    parser.add_argument('--port-rtsp', type=int, default=554, help='RTSP端口')
    parser.add_argument('--channel', type=int, default=1, help='视频通道号')
    parser.add_argument('--subtype', type=int, default=0, help='码流类型 (0=主码流, 1=辅码流1)')
    parser.add_argument('--username', type=str, default='admin', help='摄像头用户名')
    parser.add_argument('--password', type=str, default='Admin123', help='摄像头密码')
    parser.add_argument('--weights', type=str, default='./best.pt', help='模型权重')
    parser.add_argument('--device', default='cpu', help='cuda device or cpu')
    parser.add_argument('--port', type=int, default=8888, help='Web服务端口')
    args = parser.parse_args()
    
    # 构建RTSP URL（根据大华摄像头官方文档）
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.port_rtsp}/cam/realmonitor?channel={args.channel}&subtype={args.subtype}"
    
    print("=" * 60)
    print("🔥 YOLOv5 火灾烟雾检测系统")
    print("=" * 60)
    print(f"📷 摄像头地址: {args.ip}:{args.port_rtsp}")
    print(f"📡 RTSP URL: {rtsp_url}")
    print(f"🎬 通道: {args.channel}, 码流类型: {args.subtype}")
    print("=" * 60)
    
    print("正在加载模型...")
    load_model(args.weights, args.device)
    print("✅ 模型已加载")
    
    # 启动流捕获线程
    print(f"正在连接到流...")
    stream_thread = threading.Thread(target=capture_stream, args=(rtsp_url,), daemon=True)
    stream_thread.start()
    
    # 启动Flask服务
    print(f"✅ Web服务启动在 http://localhost:{args.port}")
    print(f"请在浏览器中打开: http://localhost:{args.port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=args.port, debug=False)
