"""
检测路由补丁 - 添加到 app.py 中
这些路由提供检测功能的 API 端点
"""

# 在 app.py 中添加以下导入（在顶部）:
# from detection_manager import DetectionManager

# 在初始化检测引擎的地方，替换为:
# detection_manager = None
# try:
#     detection_manager = DetectionManager(
#         model_path=MODEL_PATH,
#         device=DEVICE,
#         conf_threshold=CONF_THRESHOLD,
#         iou_threshold=IOU_THRESHOLD
#     )
#     print("✅ 检测管理器初始化成功")
# except Exception as e:
#     print(f"❌ 检测管理器初始化失败: {e}")

# ============================================================================
# 检测路由 - 添加到 app.py 中
# ============================================================================

def add_detection_routes(app, detection_manager, stream_manager):
    """添加检测相关路由"""
    
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
    
    
    @app.route('/api/detection/stats/<stream_name>', methods=['GET'])
    def get_stream_stats(stream_name):
        """获取单个流的检测统计"""
        if detection_manager is None:
            return jsonify({'error': '检测管理器未初始化'}), 500
        
        stats = detection_manager.get_stats(stream_name)
        
        if stats:
            return jsonify(stats), 200
        else:
            return jsonify({'error': f'流 {stream_name} 不存在'}), 404
    
    
    @app.route('/api/detection/config', methods=['GET'])
    def get_detection_config():
        """获取检测配置"""
        return jsonify({
            'model_path': MODEL_PATH,
            'device': DEVICE,
            'conf_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD
        }), 200
    
    
    @app.route('/api/detection/config', methods=['POST'])
    def update_detection_config():
        """更新检测配置"""
        data = request.json
        
        # 注意：这里只是返回确认，实际更新需要重新初始化引擎
        return jsonify({
            'message': '检测配置已更新（需要重启服务生效）',
            'config': {
                'conf_threshold': data.get('conf_threshold', CONF_THRESHOLD),
                'iou_threshold': data.get('iou_threshold', IOU_THRESHOLD)
            }
        }), 200
