"""
RTSP 客户端实现
处理 RTSP 协议通信，支持 RTP over UDP 和 RTP over RTSP
"""
import socket
import re
import time
import hashlib
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class RTSPMethod(Enum):
    """RTSP 方法"""
    DESCRIBE = "DESCRIBE"
    SETUP = "SETUP"
    PLAY = "PLAY"
    PAUSE = "PAUSE"
    TEARDOWN = "TEARDOWN"


@dataclass
class RTSPSession:
    """RTSP 会话信息"""
    session_id: str
    timeout: int
    server_port_rtp: int
    server_port_rtcp: int


@dataclass
class MediaDescription:
    """媒体描述信息"""
    content_type: str
    content_length: int
    sdp_content: str
    track_id: str = "0"


class RTSPClient:
    """RTSP 客户端"""
    
    def __init__(self, rtsp_url: str, username: Optional[str] = None, 
                 password: Optional[str] = None, timeout: int = 10):
        """
        初始化 RTSP 客户端
        
        Args:
            rtsp_url: RTSP URL (e.g., rtsp://192.168.1.108:554/cam/realmonitor?channel=1&subtype=0)
            username: 用户名（可选）
            password: 密码（可选）
            timeout: 连接超时时间
        """
        self.rtsp_url = rtsp_url
        self.username = username
        self.password = password
        self.timeout = timeout
        self.cseq = 0
        self.session: Optional[RTSPSession] = None
        self.socket: Optional[socket.socket] = None
        self.rtp_socket: Optional[socket.socket] = None
        self.rtcp_socket: Optional[socket.socket] = None
        self.user_agent = "LibVLC/3.0.5"
        
        # 解析 URL
        self._parse_url()
    
    def _parse_url(self):
        """解析 RTSP URL"""
        # 支持带认证信息的 URL: rtsp://user:pass@host:port/path
        # 或不带认证: rtsp://host:port/path
        match = re.match(
            r'rtsp://(?:([^:@]+)(?::([^@]+))?@)?([^:/@]+)(?::(\d+))?(/.*)',
            self.rtsp_url
        )
        if not match:
            raise ValueError(f"Invalid RTSP URL: {self.rtsp_url}")
        
        # 提取 URL 中的认证信息（如果有）
        url_user = match.group(1)
        url_pass = match.group(2)
        
        # 如果 URL 中有认证信息，使用 URL 中的；否则使用构造函数参数
        if url_user:
            self.username = url_user
        if url_pass:
            self.password = url_pass
        
        self.host = match.group(3)
        self.port = int(match.group(4)) if match.group(4) else 554
        self.path = match.group(5)
        
        # 构建不含认证信息的 URL（用于 RTSP 请求）
        self.request_url = f"rtsp://{self.host}:{self.port}{self.path}"
    
    def connect(self) -> bool:
        """建立 RTSP 连接"""
        try:
            print(f"[RTSP] Connecting to {self.host}:{self.port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            print(f"[RTSP] Connected successfully")
            return True
        except Exception as e:
            print(f"[RTSP] Failed to connect to {self.host}:{self.port}: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def _get_next_cseq(self) -> int:
        """获取下一个 CSeq"""
        self.cseq += 1
        return self.cseq
    
    def _parse_digest_auth(self, www_authenticate: str) -> Dict[str, str]:
        """解析 Digest 认证头"""
        auth_dict = {}
        # 移除 "Digest " 前缀
        if www_authenticate.startswith('Digest '):
            www_authenticate = www_authenticate[7:]
        
        # 提取 Digest 参数 - 需要正确处理带引号的值
        import re as regex
        pattern = r'(\w+)=(?:"([^"]*)"|([^,\s]*))'
        matches = regex.findall(pattern, www_authenticate)
        
        for key, quoted_value, unquoted_value in matches:
            value = quoted_value if quoted_value else unquoted_value
            auth_dict[key] = value
        
        print(f"[RTSP] Parsed auth dict: {auth_dict}")
        return auth_dict
    
    def _calculate_digest_response(self, method: str, uri: str, 
                                   auth_dict: Dict[str, str], 
                                   nonce_count: int = 1) -> str:
        """计算 Digest 认证响应"""
        if not self.username or not self.password:
            return ""
        
        realm = auth_dict.get('realm', '')
        nonce = auth_dict.get('nonce', '')
        
        # 计算 HA1 = MD5(username:realm:password)
        ha1_str = f"{self.username}:{realm}:{self.password}"
        ha1 = hashlib.md5(ha1_str.encode()).hexdigest()
        
        # 计算 HA2 = MD5(method:uri)
        ha2_str = f"{method}:{uri}"
        ha2 = hashlib.md5(ha2_str.encode()).hexdigest()
        
        # 计算 response = MD5(HA1:nonce:HA2)
        response_str = f"{ha1}:{nonce}:{ha2}"
        response = hashlib.md5(response_str.encode()).hexdigest()
        
        return response
    
    def _build_digest_auth_header(self, method: str, uri: str, 
                                  www_authenticate: str) -> str:
        """构建 Digest 认证头"""
        auth_dict = self._parse_digest_auth(www_authenticate)
        response = self._calculate_digest_response(method, uri, auth_dict)
        
        realm = auth_dict.get('realm', '')
        nonce = auth_dict.get('nonce', '')
        
        auth_header = (
            f'Digest username="{self.username}", '
            f'realm="{realm}", '
            f'nonce="{nonce}", '
            f'uri="{uri}", '
            f'response="{response}"'
        )
        return auth_header
    
    def _build_request(self, method: RTSPMethod, url: str, 
                      headers: Optional[Dict[str, str]] = None) -> str:
        """构建 RTSP 请求"""
        request = f"{method.value} {url} RTSP/1.0\r\n"
        request += f"CSeq: {self._get_next_cseq()}\r\n"
        request += f"User-Agent: {self.user_agent}\r\n"
        
        if headers:
            for key, value in headers.items():
                request += f"{key}: {value}\r\n"
        
        request += "\r\n"
        return request
    
    def _send_request(self, request: str) -> str:
        """发送 RTSP 请求并获取响应"""
        if not self.socket:
            raise RuntimeError("Not connected to RTSP server")
        
        self.socket.sendall(request.encode())
        
        # 接收响应
        response = b""
        while True:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                response += chunk
                # 简单的响应完整性检查
                if b"\r\n\r\n" in response:
                    break
            except socket.timeout:
                break
        
        return response.decode('utf-8', errors='ignore')
    
    def describe(self) -> Optional[MediaDescription]:
        """
        DESCRIBE 命令：获取媒体描述
        """
        print(f"[RTSP] Sending DESCRIBE to {self.request_url}")
        request = self._build_request(RTSPMethod.DESCRIBE, self.request_url)
        response = self._send_request(request)
        
        # 处理 401 认证失败
        if "401 Unauthorized" in response:
            print(f"[RTSP] Got 401 Unauthorized, retrying with Digest auth")
            # 提取 WWW-Authenticate 头
            www_auth = None
            for line in response.split('\r\n'):
                if line.startswith('WWW-Authenticate:'):
                    www_auth = line.split(': ', 1)[1]
                    print(f"[RTSP] WWW-Authenticate: {www_auth}")
                    break
            
            if www_auth:
                # 构建带认证的请求
                auth_header = self._build_digest_auth_header(
                    "DESCRIBE", self.request_url, www_auth
                )
                print(f"[RTSP] Authorization: {auth_header}")
                request = self._build_request(RTSPMethod.DESCRIBE, self.request_url)
                # 手动添加认证头
                request = request.replace("\r\n\r\n", f"\r\nAuthorization: {auth_header}\r\n\r\n")
                print(f"[RTSP] Retrying DESCRIBE with auth...")
                response = self._send_request(request)
            else:
                print(f"[RTSP] No WWW-Authenticate header found")
        
        if "200 OK" not in response:
            print(f"[RTSP] DESCRIBE failed: {response[:300]}")
            return None
        
        print(f"[RTSP] DESCRIBE successful")
        
        # 解析响应头
        headers = {}
        lines = response.split('\r\n')
        sdp_start = -1
        
        for i, line in enumerate(lines):
            if ': ' in line:
                key, value = line.split(': ', 1)
                headers[key] = value
            elif line == '' and sdp_start == -1:
                sdp_start = i + 1
                break
        
        # 获取 SDP 内容
        sdp_content = '\r\n'.join(lines[sdp_start:]) if sdp_start > 0 else ""
        
        return MediaDescription(
            content_type=headers.get('Content-Type', ''),
            content_length=int(headers.get('Content-Length', 0)),
            sdp_content=sdp_content,
            track_id="0"
        )
    
    def setup(self, client_port_rtp: int = 63088, 
              client_port_rtcp: int = 63089) -> Optional[RTSPSession]:
        """
        SETUP 命令：建立传输通道
        
        Args:
            client_port_rtp: 客户端 RTP 端口
            client_port_rtcp: 客户端 RTCP 端口
        """
        # 创建 UDP socket
        try:
            self.rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rtp_socket.bind(('0.0.0.0', client_port_rtp))
            
            self.rtcp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rtcp_socket.bind(('0.0.0.0', client_port_rtcp))
        except Exception as e:
            print(f"Failed to create UDP sockets: {e}")
            return None
        
        # 构建 SETUP 请求
        setup_url = f"{self.request_url}/trackID=0"
        headers = {
            "Transport": f"RTP/AVP;unicast;client_port={client_port_rtp}-{client_port_rtcp}"
        }
        
        print(f"[RTSP] Sending SETUP to {setup_url}")
        request = self._build_request(RTSPMethod.SETUP, setup_url, headers)
        response = self._send_request(request)
        
        # 处理 401 认证失败
        if "401 Unauthorized" in response:
            print(f"[RTSP] Got 401 Unauthorized in SETUP, retrying with Digest auth")
            # 提取 WWW-Authenticate 头
            www_auth = None
            for line in response.split('\r\n'):
                if line.startswith('WWW-Authenticate:'):
                    www_auth = line.split(': ', 1)[1]
                    break
            
            if www_auth:
                # 构建带认证的请求
                auth_header = self._build_digest_auth_header(
                    "SETUP", setup_url, www_auth
                )
                request = self._build_request(RTSPMethod.SETUP, setup_url, headers)
                # 手动添加认证头
                request = request.replace("\r\n\r\n", f"\r\nAuthorization: {auth_header}\r\n\r\n")
                response = self._send_request(request)
        
        if "200 OK" not in response:
            print(f"[RTSP] SETUP failed: {response[:300]}")
            return None
        
        print(f"[RTSP] SETUP successful")
        
        # 解析响应
        session_id = None
        timeout = 60
        server_port_rtp = 0
        server_port_rtcp = 0
        
        for line in response.split('\r\n'):
            if line.startswith('Session:'):
                session_id = line.split(': ')[1].split(';')[0]
            elif line.startswith('Transport:'):
                # 解析 server_port
                match = re.search(r'server_port=(\d+)-(\d+)', line)
                if match:
                    server_port_rtp = int(match.group(1))
                    server_port_rtcp = int(match.group(2))
        
        if not session_id:
            print("No session ID in SETUP response")
            return None
        
        self.session = RTSPSession(
            session_id=session_id,
            timeout=timeout,
            server_port_rtp=server_port_rtp,
            server_port_rtcp=server_port_rtcp
        )
        
        return self.session
    
    def play(self) -> bool:
        """
        PLAY 命令：开始播放
        """
        if not self.session:
            print("[RTSP] No active session. Call setup() first.")
            return False
        
        play_url = f"{self.request_url}/"
        headers = {
            "Session": self.session.session_id,
            "Range": "npt=0.000-"
        }
        
        print(f"[RTSP] Sending PLAY to {play_url}")
        request = self._build_request(RTSPMethod.PLAY, play_url, headers)
        response = self._send_request(request)
        
        # 处理 401 认证失败
        if "401 Unauthorized" in response:
            print(f"[RTSP] Got 401 Unauthorized in PLAY, retrying with Digest auth")
            # 提取 WWW-Authenticate 头
            www_auth = None
            for line in response.split('\r\n'):
                if line.startswith('WWW-Authenticate:'):
                    www_auth = line.split(': ', 1)[1]
                    break
            
            if www_auth:
                # 构建带认证的请求
                auth_header = self._build_digest_auth_header(
                    "PLAY", play_url, www_auth
                )
                request = self._build_request(RTSPMethod.PLAY, play_url, headers)
                # 手动添加认证头
                request = request.replace("\r\n\r\n", f"\r\nAuthorization: {auth_header}\r\n\r\n")
                response = self._send_request(request)
        
        if "200 OK" not in response:
            print(f"[RTSP] PLAY failed: {response[:300]}")
            return False
        
        print(f"[RTSP] PLAY successful")
        return True
    
    def pause(self) -> bool:
        """
        PAUSE 命令：暂停播放
        """
        if not self.session:
            print("No active session.")
            return False
        
        pause_url = f"{self.request_url}/"
        headers = {
            "Session": self.session.session_id
        }
        
        request = self._build_request(RTSPMethod.PAUSE, pause_url, headers)
        response = self._send_request(request)
        
        return "200 OK" in response
    
    def teardown(self) -> bool:
        """
        TEARDOWN 命令：停止播放
        """
        if not self.session:
            print("No active session.")
            return False
        
        teardown_url = f"{self.request_url}/"
        headers = {
            "Session": self.session.session_id
        }
        
        request = self._build_request(RTSPMethod.TEARDOWN, teardown_url, headers)
        response = self._send_request(request)
        
        # 关闭 UDP socket
        if self.rtp_socket:
            self.rtp_socket.close()
            self.rtp_socket = None
        if self.rtcp_socket:
            self.rtcp_socket.close()
            self.rtcp_socket = None
        
        self.session = None
        
        return "200 OK" in response
    
    def get_rtp_socket(self) -> Optional[socket.socket]:
        """获取 RTP socket"""
        return self.rtp_socket
    
    def get_server_address(self) -> Tuple[str, int]:
        """获取服务器地址"""
        if not self.session:
            raise RuntimeError("No active session")
        return (self.host, self.session.server_port_rtp)
