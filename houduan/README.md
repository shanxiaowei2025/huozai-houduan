# RTSP 实时媒体流服务器

基于 Python 的 RTSP 协议实现，支持实时获取摄像头媒体流。

## 功能特性

- ✅ 完整的 RTSP 协议实现（DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN）
- ✅ RTP over UDP 传输支持
- ✅ 多路并发流管理
- ✅ RESTful API 接口
- ✅ 实时统计信息
- ✅ H.264 视频帧解析

## 项目结构

```
houduan/
├── app.py                 # Flask API 服务器
├── rtsp_client.py        # RTSP 客户端实现
├── media_stream.py       # 媒体流处理（RTP 解析、帧缓冲）
├── stream_manager.py     # 流管理器
├── requirements.txt      # 依赖包
├── config.json          # 流配置文件
├── .env                 # 环境变量配置
└── README.md           # 本文件
```

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：如需视频解码功能（可选），请额外安装：
```bash
brew install pkg-config ffmpeg
pip install av opencv-python
```

### 2. 配置环境变量

编辑 `.env` 文件：

```
HOST=0.0.0.0
PORT=5001
DEBUG=False
RTSP_TIMEOUT=10
```

### 3. 配置流

编辑 `config.json` 添加摄像头信息：

```json
{
  "streams": [
    {
      "name": "camera_1",
      "rtsp_url": "rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0",
      "channel": 1,
      "subtype": 0,
      "username": null,
      "password": null,
      "enabled": true
    }
  ]
}
```

## 使用

### 启动服务器

```bash
python app.py
```

服务器将在 `http://0.0.0.0:5000` 启动

### API 端点

#### 1. 健康检查

```bash
GET /api/health
```

响应：
```json
{
  "status": "ok",
  "message": "RTSP Media Stream Server is running"
}
```

# 2. 启动流
curl -X POST http://localhost:5001/api/streams/camera_1/start

# 3. 查看实时状态
curl http://localhost:5001/api/streams/camera_1


#### 2. 列出所有流

```bash
GET /api/streams
```

响应：
```json
{
  "total": 2,
  "streams": [
    {
      "name": "camera_1",
      "connected": true,
      "playing": true,
      "packets_received": 1250,
      "frames_received": 42,
      "bytes_received": 512000
    }
  ]
}
```

#### 3. 添加新流

```bash
POST /api/streams
Content-Type: application/json

{
  "name": "camera_1",
  "rtsp_url": "rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0",
  "channel": 1,
  "subtype": 0,
  "username": null,
  "password": null,
  "enabled": true
}
```

#### 4. 获取流状态

```bash
GET /api/streams/{name}
```

响应：
```json
{
  "name": "camera_1",
  "connected": true,
  "playing": true,
  "packets_received": 1250,
  "frames_received": 42,
  "bytes_received": 512000,
  "last_frame_time": 1700000000.123
}
```

#### 5. 启动流

```bash
POST /api/streams/{name}/start
```

#### 6. 停止流

```bash
POST /api/streams/{name}/stop
```

#### 7. 删除流

```bash
DELETE /api/streams/{name}
```

#### 8. 获取最新帧

```bash
GET /api/streams/{name}/frame
```

返回原始 H.264 帧数据

#### 9. 从配置文件加载流

```bash
POST /api/config/streams
Content-Type: application/json

{
  "streams": [
    {
      "name": "camera_1",
      "rtsp_url": "rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0",
      "channel": 1,
      "subtype": 0
    }
  ]
}
```

## 工作流程

### RTSP 连接流程

1. **DESCRIBE** - 获取媒体描述（SDP）
2. **SETUP** - 建立 RTP/UDP 传输通道
3. **PLAY** - 开始接收媒体流
4. **TEARDOWN** - 关闭连接

### 数据流处理

```
RTSP Server
    ↓
RTP/UDP Packets
    ↓
RTPParser (解析 RTP 头)
    ↓
H264FrameBuffer (缓冲帧数据)
    ↓
VideoFrameDecoder (解码视频帧)
    ↓
API Response
```

## 技术细节

### RTSP 协议

- 基于 RFC 2326 标准
- 支持 RTP over UDP 传输
- 支持 HTTP Digest 认证（预留）

### RTP 数据包结构

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|X|  CC   |M|     PT      |       sequence number         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           timestamp                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           synchronization source (SSRC) identifier            |
+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
|            contributing source (CSRC) identifiers             |
|                             ....                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### H.264 帧识别

- 通过 RTP marker bit 识别帧边界
- 支持多个 RTP 包组成一个完整帧

## 性能优化

- 多线程接收和处理
- 环形缓冲区管理帧数据
- 异步 API 响应

## 故障排查

### 连接失败

1. 检查 RTSP URL 是否正确
2. 确认摄像头设备在线
3. 检查防火墙设置（端口 554）
4. 验证用户名和密码

### 无法接收数据

1. 检查 UDP 端口是否被占用
2. 确认网络连接正常
3. 查看服务器日志

### 帧解码失败

1. 确认摄像头支持 H.264 编码
2. 检查 PyAV 库是否正确安装
3. 验证 FFmpeg 库可用

## 扩展功能

### 支持的功能（已实现）

- [x] RTSP 基本协议
- [x] RTP 数据包解析
- [x] 多流管理
- [x] RESTful API
- [x] 统计信息

### 待实现功能

- [ ] HTTP Digest 认证
- [ ] RTCP 反馈
- [ ] 动态比特率调整
- [ ] 视频录制
- [ ] WebSocket 实时推流
- [ ] 多编码格式支持（H.265, VP9）

## 依赖说明

| 包名 | 版本 | 用途 |
|------|------|------|
| Flask | 2.3.3 | Web 框架 |
| Flask-CORS | 4.0.0 | 跨域资源共享 |
| opencv-python | 4.8.1.78 | 视频处理 |
| av | 10.0.0 | 媒体解码 |
| numpy | 1.24.3 | 数值计算 |
| requests | 2.31.0 | HTTP 请求 |
| python-dotenv | 1.0.0 | 环境变量管理 |

## 示例代码

### Python 客户端

```python
import requests
import json

BASE_URL = "http://localhost:5000/api"

# 添加流
response = requests.post(f"{BASE_URL}/streams", json={
    "name": "camera_1",
    "rtsp_url": "rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0",
    "channel": 1,
    "subtype": 0
})
print(response.json())

# 启动流
response = requests.post(f"{BASE_URL}/streams/camera_1/start")
print(response.json())

# 获取状态
response = requests.get(f"{BASE_URL}/streams/camera_1")
print(response.json())

# 获取帧
response = requests.get(f"{BASE_URL}/streams/camera_1/frame")
with open("frame.h264", "wb") as f:
    f.write(response.content)

# 停止流
response = requests.post(f"{BASE_URL}/streams/camera_1/stop")
print(response.json())
```

### cURL 示例

```bash
# 添加流
curl -X POST http://localhost:5000/api/streams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "camera_1",
    "rtsp_url": "rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0",
    "channel": 1,
    "subtype": 0
  }'

# 启动流
curl -X POST http://localhost:5000/api/streams/camera_1/start

# 获取状态
curl http://localhost:5000/api/streams/camera_1

# 停止流
curl -X POST http://localhost:5000/api/streams/camera_1/stop
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
