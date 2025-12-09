"""
媒体流处理模块
处理 RTP 数据接收、解析和视频帧提取
"""
import socket
import struct
import threading
import time
from typing import Optional, Callable
from collections import deque
from dataclasses import dataclass


@dataclass
class RTPPacket:
    """RTP 数据包"""
    version: int
    padding: bool
    extension: bool
    cc: int
    marker: bool
    payload_type: int
    sequence_number: int
    timestamp: int
    ssrc: int
    payload: bytes


class RTPParser:
    """RTP 数据包解析器"""
    
    @staticmethod
    def parse(data: bytes) -> Optional[RTPPacket]:
        """解析 RTP 数据包"""
        if len(data) < 12:
            return None
        
        # 解析 RTP 头 (12 字节固定)
        byte0 = data[0]
        version = (byte0 >> 6) & 0x3
        padding = bool((byte0 >> 5) & 0x1)
        extension = bool((byte0 >> 4) & 0x1)
        cc = byte0 & 0xf
        
        byte1 = data[1]
        marker = bool((byte1 >> 7) & 0x1)
        payload_type = byte1 & 0x7f
        
        sequence_number = struct.unpack('!H', data[2:4])[0]
        timestamp = struct.unpack('!I', data[4:8])[0]
        ssrc = struct.unpack('!I', data[8:12])[0]
        
        # 计算 payload 起始位置
        payload_start = 12 + cc * 4
        
        if len(data) < payload_start:
            return None
        
        payload = data[payload_start:]
        
        return RTPPacket(
            version=version,
            padding=padding,
            extension=extension,
            cc=cc,
            marker=marker,
            payload_type=payload_type,
            sequence_number=sequence_number,
            timestamp=timestamp,
            ssrc=ssrc,
            payload=payload
        )


class H264FrameBuffer:
    """H.264/H.265 帧缓冲区"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.current_frame = b""
        self.last_timestamp = None
    
    def add_packet(self, packet: RTPPacket):
        """添加 RTP 数据包"""
        with self.lock:
            self.buffer.append(packet)
    
    def get_frame(self) -> Optional[bytes]:
        """获取完整的 H.265/H.264 帧"""
        with self.lock:
            if not self.buffer:
                return None
            
            # 收集所有包直到找到 marker bit
            frame_data = b""
            frame_found = False
            
            while self.buffer:
                packet = self.buffer.popleft()
                
                # 为每个 NAL 单元添加起始码（H.265/H.264 需要）
                # 起始码格式：0x00 0x00 0x00 0x01
                if len(packet.payload) > 0:
                    frame_data += b"\x00\x00\x00\x01" + packet.payload
                
                # marker bit = 1 表示这是一个完整帧的最后一个包
                if packet.marker:
                    frame_found = True
                    break
            
            # 只返回找到完整帧的数据
            if frame_found and len(frame_data) > 4:  # 至少要有起始码
                return frame_data
            
            return None
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
            self.current_frame = b""


class MediaStreamReceiver:
    """媒体流接收器"""
    
    def __init__(self, rtp_socket: socket.socket, server_address: tuple,
                 on_frame: Optional[Callable[[bytes], None]] = None):
        """
        初始化媒体流接收器
        
        Args:
            rtp_socket: RTP UDP socket
            server_address: 服务器地址 (host, port)
            on_frame: 帧回调函数
        """
        self.rtp_socket = rtp_socket
        self.server_address = server_address
        self.on_frame = on_frame
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_buffer = H264FrameBuffer()
        self.stats = {
            'packets_received': 0,
            'frames_received': 0,
            'bytes_received': 0
        }
    
    def start(self):
        """启动接收"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止接收"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _receive_loop(self):
        """接收循环"""
        self.rtp_socket.settimeout(2.0)
        
        while self.running:
            try:
                data, addr = self.rtp_socket.recvfrom(65536)
                self.stats['packets_received'] += 1
                self.stats['bytes_received'] += len(data)
                
                # 解析 RTP 数据包
                packet = RTPParser.parse(data)
                if packet:
                    self.frame_buffer.add_packet(packet)
                    
                    # 如果是帧的最后一个包，尝试提取完整帧
                    if packet.marker:
                        frame_data = self.frame_buffer.get_frame()
                        if frame_data and self.on_frame:
                            self.stats['frames_received'] += 1
                            self.on_frame(frame_data)
            
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error in receive loop: {e}")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()


class VideoFrameDecoder:
    """视频帧缓冲器
    
    保存原始 H.264 帧数据。如需解码为图像，可选安装：
    brew install pkg-config ffmpeg
    pip install av opencv-python
    """
    
    def __init__(self):
        self.frame_buffer = deque(maxlen=30)  # 保留最近 30 帧
        self.lock = threading.Lock()
    
    def decode_h264_frame(self, frame_data: bytes) -> Optional[bytes]:
        """返回原始 H.264 帧数据"""
        return frame_data
    
    def add_frame(self, frame: bytes):
        """添加帧数据"""
        with self.lock:
            self.frame_buffer.append(frame)
    
    def get_latest_frame(self) -> Optional[bytes]:
        """获取最新的帧数据"""
        with self.lock:
            if self.frame_buffer:
                return self.frame_buffer[-1]
            return None
    
    def get_frame_count(self) -> int:
        """获取缓冲区中的帧数"""
        with self.lock:
            return len(self.frame_buffer)
