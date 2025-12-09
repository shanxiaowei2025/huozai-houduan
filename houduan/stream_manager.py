"""
流管理器
管理多个 RTSP 连接和媒体流
"""
import threading
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from rtsp_client import RTSPClient
from media_stream import MediaStreamReceiver, VideoFrameDecoder


@dataclass
class StreamConfig:
    """流配置"""
    name: str
    rtsp_url: str
    channel: int
    subtype: int
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: bool = True


@dataclass
class StreamStatus:
    """流状态"""
    name: str
    connected: bool
    playing: bool
    packets_received: int
    frames_received: int
    bytes_received: int
    last_frame_time: float


class StreamManager:
    """流管理器"""
    
    def __init__(self):
        self.streams: Dict[str, 'ManagedStream'] = {}
        self.lock = threading.Lock()
    
    def add_stream(self, config: StreamConfig) -> bool:
        """添加流"""
        with self.lock:
            if config.name in self.streams:
                print(f"Stream {config.name} already exists")
                return False
            
            stream = ManagedStream(config)
            self.streams[config.name] = stream
            return True
    
    def remove_stream(self, name: str) -> bool:
        """移除流"""
        with self.lock:
            if name not in self.streams:
                return False
            
            stream = self.streams.pop(name)
            stream.stop()
            return True
    
    def start_stream(self, name: str) -> bool:
        """启动流"""
        with self.lock:
            if name not in self.streams:
                return False
            
            return self.streams[name].start()
    
    def stop_stream(self, name: str) -> bool:
        """停止流"""
        with self.lock:
            if name not in self.streams:
                return False
            
            return self.streams[name].stop()
    
    def get_status(self, name: str) -> Optional[StreamStatus]:
        """获取流状态"""
        with self.lock:
            if name not in self.streams:
                return None
            
            return self.streams[name].get_status()
    
    def get_all_status(self) -> List[StreamStatus]:
        """获取所有流状态"""
        with self.lock:
            return [stream.get_status() for stream in self.streams.values()]
    
    def get_frame(self, name: str) -> Optional[bytes]:
        """获取流的最新帧"""
        with self.lock:
            if name not in self.streams:
                return None
            
            return self.streams[name].get_frame()
    
    
    def list_streams(self) -> List[str]:
        """列出所有流"""
        with self.lock:
            return list(self.streams.keys())


class ManagedStream:
    """受管理的流"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.rtsp_client: Optional[RTSPClient] = None
        self.receiver: Optional[MediaStreamReceiver] = None
        self.decoder = VideoFrameDecoder()
        self.running = False
        self.connected = False
        self.playing = False
        self.last_frame_time = 0
        self.sps_pps_data = b""  # 存储 SPS/PPS 参数集
    
    def start(self) -> bool:
        """启动流"""
        if self.running:
            return True
        
        self.running = True
        thread = threading.Thread(target=self._connect_and_play, daemon=True)
        thread.start()
        
        return True
    
    def stop(self) -> bool:
        """停止流"""
        self.running = False
        
        if self.playing:
            try:
                self.rtsp_client.pause()
                self.rtsp_client.teardown()
            except:
                pass
            self.playing = False
        
        if self.receiver:
            self.receiver.stop()
        
        if self.rtsp_client:
            self.rtsp_client.disconnect()
        
        self.connected = False
        return True
    
    
    def _extract_sps_pps_from_sdp(self, sdp_content: str) -> bytes:
        """从 SDP 中提取 SPS/PPS 参数集"""
        import base64
        import re
        
        sps_pps = b""
        
        # 查找 sprop-parameter-sets（H.265 使用）
        match = re.search(r'sprop-parameter-sets=([^,\s;]+)', sdp_content)
        if match:
            params_str = match.group(1)
            # 参数集以逗号分隔
            param_sets = params_str.split(',')
            for param in param_sets:
                try:
                    # Base64 解码
                    decoded = base64.b64decode(param)
                    # 添加 NAL 起始码
                    sps_pps += b"\x00\x00\x00\x01" + decoded
                    print(f"[SDP] 提取参数集: {len(decoded)} 字节")
                except Exception as e:
                    print(f"[SDP] 参数集解码失败: {e}")
        
        # 如果没有找到 H.265 参数集，尝试查找 H.264 参数集
        if not sps_pps:
            match = re.search(r'sprop-parameter-sets=([^,\s;]+),([^,\s;]+)', sdp_content)
            if match:
                for i in range(1, 3):
                    try:
                        decoded = base64.b64decode(match.group(i))
                        sps_pps += b"\x00\x00\x00\x01" + decoded
                        print(f"[SDP] 提取 H.264 参数集 {i}: {len(decoded)} 字节")
                    except Exception as e:
                        print(f"[SDP] H.264 参数集 {i} 解码失败: {e}")
        
        return sps_pps
    
    def _connect_and_play(self):
        """连接并播放"""
        try:
            # 创建 RTSP 客户端
            self.rtsp_client = RTSPClient(
                self.config.rtsp_url,
                username=self.config.username,
                password=self.config.password
            )
            
            # 连接
            if not self.rtsp_client.connect():
                print(f"Failed to connect to {self.config.name}")
                return
            
            self.connected = True
            print(f"Connected to {self.config.name}")
            
            # DESCRIBE
            media_desc = self.rtsp_client.describe()
            if not media_desc:
                print(f"DESCRIBE failed for {self.config.name}")
                return
            
            print(f"Media description for {self.config.name}: {media_desc.content_type}")
            
            # 从 SDP 中提取参数集
            self.sps_pps_data = self._extract_sps_pps_from_sdp(media_desc.sdp_content)
            print(f"[SDP] 总参数集大小: {len(self.sps_pps_data)} 字节")
            
            # SETUP
            session = self.rtsp_client.setup()
            if not session:
                print(f"SETUP failed for {self.config.name}")
                return
            
            print(f"Session established for {self.config.name}: {session.session_id}")
            
            # 创建接收器
            def on_frame(frame_data: bytes):
                self.last_frame_time = time.time()
                # 将帧数据添加到解码器缓冲区
                self.decoder.add_frame(frame_data)
            
            self.receiver = MediaStreamReceiver(
                self.rtsp_client.get_rtp_socket(),
                self.rtsp_client.get_server_address(),
                on_frame=on_frame
            )
            self.receiver.start()
            
            # PLAY
            if not self.rtsp_client.play():
                print(f"PLAY failed for {self.config.name}")
                return
            
            self.playing = True
            print(f"Playing {self.config.name}")
            
            # 保持连接
            while self.running:
                time.sleep(1)
        
        except Exception as e:
            print(f"Error in stream {self.config.name}: {e}")
        
        finally:
            self.stop()
    
    def get_status(self) -> StreamStatus:
        """获取状态"""
        stats = self.receiver.get_stats() if self.receiver else {}
        
        return StreamStatus(
            name=self.config.name,
            connected=self.connected,
            playing=self.playing,
            packets_received=stats.get('packets_received', 0),
            frames_received=stats.get('frames_received', 0),
            bytes_received=stats.get('bytes_received', 0),
            last_frame_time=self.last_frame_time
        )
    
    def get_frame(self) -> Optional[bytes]:
        """获取最新帧"""
        # 从解码器缓冲区获取最新帧
        frame_data = self.decoder.get_latest_frame()
        if frame_data and self.sps_pps_data:
            # 在帧数据前添加 SPS/PPS 参数集
            return self.sps_pps_data + frame_data
        return frame_data
    
    
