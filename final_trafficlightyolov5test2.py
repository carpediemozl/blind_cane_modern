import cv2
import depthai as dai
import numpy as np
import time
from collections import deque, Counter

# --- 1. 核心参数 ---
MODEL_PATH = 'models/knottedblob/best10301nohue_openvino_2022.1_6shave.blob'
LABEL_MAP = ['Green', 'Red', 'Yellow']
NN_INPUT_SIZE = 416

# === 高级过滤参数 ===
CONF_THRESHOLD = 0.4  # 全局阈值（降低以捕获更多候选）

# 类别特定阈值（根据数据集不平衡调整）
CLASS_THRESHOLDS = {
    'Green': 0.55,   # 提高绿灯阈值（因为样本多，容易过拟合）
    'Red': 0.50,     # 红灯正常
    'Yellow': 0.45   # 降低黄灯阈值（因为样本少，否则会漏检）
}

# 时序平滑参数
USE_TEMPORAL_SMOOTHING = True  # 启用时序平滑
BUFFER_SIZE = 7                # 保留最近7帧
MIN_VOTES = 4                  # 至少4帧同意才显示

# 显示选项
SHOW_ONLY_BEST = True
DEBUG_MODE = True
SHOW_ALL_CANDIDATES = False  # 是否显示所有候选检测（调试用）

# --- 2. 时序平滑类 ---
class TemporalSmoother:
    """时序平滑器：减少检测抖动和误检"""
    def __init__(self, buffer_size=7, min_votes=4):
        self.buffer_size = buffer_size
        self.min_votes = min_votes
        self.label_buffer = deque(maxlen=buffer_size)
        self.conf_buffer = deque(maxlen=buffer_size)
        self.bbox_buffer = deque(maxlen=buffer_size)
        
    def update(self, detection):
        """更新缓冲区"""
        if detection is not None:
            self.label_buffer.append(detection.label)
            self.conf_buffer.append(detection.confidence)
            bbox = [detection.xmin, detection.ymin, detection.xmax, detection.ymax]
            self.bbox_buffer.append(bbox)
        else:
            # 无检测时添加 None
            self.label_buffer.append(None)
            self.conf_buffer.append(0)
            self.bbox_buffer.append(None)
    
    def get_smoothed_detection(self):
        """获取平滑后的检测结果"""
        if len(self.label_buffer) < self.min_votes:
            return None
        
        # 统计最近N帧的标签投票
        valid_labels = [l for l in self.label_buffer if l is not None]
        
        if len(valid_labels) < self.min_votes:
            return None
        
        # 投票
        label_counts = Counter(valid_labels)
        most_common_label, vote_count = label_counts.most_common(1)[0]
        
        # 需要至少 min_votes 次投票
        if vote_count < self.min_votes:
            return None
        
        # 计算该标签的平均置信度
        label_confs = [self.conf_buffer[i] for i in range(len(self.label_buffer)) 
                       if self.label_buffer[i] == most_common_label]
        avg_confidence = np.mean(label_confs) if label_confs else 0
        
        # 计算平均边界框
        label_bboxes = [self.bbox_buffer[i] for i in range(len(self.label_buffer))
                        if self.label_buffer[i] == most_common_label and self.bbox_buffer[i] is not None]
        
        if label_bboxes:
            avg_bbox = np.mean(label_bboxes, axis=0)
        else:
            avg_bbox = None
        
        return {
            'label': most_common_label,
            'confidence': avg_confidence,
            'bbox': avg_bbox,
            'votes': vote_count,
            'total_frames': len(self.label_buffer)
        }
    
    def reset(self):
        """重置缓冲区"""
        self.label_buffer.clear()
        self.conf_buffer.clear()
        self.bbox_buffer.clear()

# --- 3. 构建 Pipeline ---
print("=" * 70)
print("红绿灯检测系统 - 高级过滤版")
print("=" * 70)
print(f"数据集分布: Green={49.9}% | Red={35.5}% | Yellow={14.6}%")
print(f"类别阈值: {CLASS_THRESHOLDS}")
print(f"时序平滑: {USE_TEMPORAL_SMOOTHING} (窗口={BUFFER_SIZE}, 最小投票={MIN_VOTES})")
print("=" * 70)

pipeline = dai.Pipeline()

# 定义节点
cam_rgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")

# 配置节点
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

spatial_nn.setBlobPath(MODEL_PATH)
spatial_nn.setConfidenceThreshold(0.3)
spatial_nn.input.setBlocking(False)
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)
spatial_nn.setDepthUpperThreshold(10000)

spatial_nn.setNumClasses(len(LABEL_MAP))
spatial_nn.setCoordinateSize(4)
spatial_nn.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
spatial_nn.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
spatial_nn.setIouThreshold(0.5)

# 链接节点
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
cam_rgb.preview.link(spatial_nn.input)
stereo.depth.link(spatial_nn.inputDepth)

spatial_nn.passthrough.link(xout_rgb.input)
spatial_nn.out.link(xout_nn.input)

print("\n正在连接设备...")

# --- 4. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("✓ 设备连接成功！")
        print(f"  DepthAI 版本: {dai.__version__}\n")
        print("按键说明:")
        print("  q - 退出")
        print("  d - 切换调试模式")
        print("  t - 切换时序平滑")
        print("  c - 显示所有候选检测")
        print("  r - 重置平滑器")
        print("  1/2 - 调整绿灯阈值")
        print("  3/4 - 调整红灯阈值")
        print("  5/6 - 调整黄灯阈值")
        print("=" * 70 + "\n")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        # 初始化时序平滑器
        smoother = TemporalSmoother(BUFFER_SIZE, MIN_VOTES)
        
        startTime = time.monotonic()
        counter = 0
        fps = 0
        frame_count = 0
        
        detection_stats = {label: 0 for label in LABEL_MAP}
        smoothed_stats = {label: 0 for label in LABEL_MAP}

        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()

            frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            frame_count += 1
            
            # FPS
            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime
            
            # === 过滤原始检测 ===
            filtered_detections = []
            for det in detections:
                try:
                    label = LABEL_MAP[det.label]
                except:
                    continue
                
                threshold = CLASS_THRESHOLDS.get(label, CONF_THRESHOLD)
                
                if det.confidence >= threshold:
                    filtered_detections.append(det)
            
            # 按置信度排序
            filtered_detections = sorted(filtered_detections, key=lambda x: x.confidence, reverse=True)
            
            # 选择最佳检测
            best_detection = filtered_detections[0] if filtered_detections else None
            
            # === 时序平滑 ===
            if USE_TEMPORAL_SMOOTHING:
                smoother.update(best_detection)
                smoothed_result = smoother.get_smoothed_detection()
            else:
                smoothed_result = None
            
            # === 调试信息 ===
            if DEBUG_MODE and frame_count % 30 == 0:
                print(f"\n[帧 {frame_count}] FPS: {fps:.1f}")
                print(f"  原始检测: {len(filtered_detections)} 个")
                
                if best_detection:
                    label = LABEL_MAP[best_detection.label]
                    print(f"  最佳: {label} ({best_detection.confidence:.3f})")
                
                if smoothed_result:
                    label = LABEL_MAP[smoothed_result['label']]
                    print(f"  平滑后: {label} (avg_conf={smoothed_result['confidence']:.3f}, "
                          f"votes={smoothed_result['votes']}/{smoothed_result['total_frames']})")
                else:
                    print(f"  平滑后: 无 (等待更多帧或投票不足)")
            
            # === 绘制 ===
            # 1. 绘制所有候选（如果启用）
            if SHOW_ALL_CANDIDATES and filtered_detections:
                for i, det in enumerate(filtered_detections[:3]):  # 最多3个
                    x1 = int(det.xmin * frame.shape[1])
                    x2 = int(det.xmax * frame.shape[1])
                    y1 = int(det.ymin * frame.shape[0])
                    y2 = int(det.ymax * frame.shape[0])
                    
                    label = LABEL_MAP[det.label]
                    
                    # 半透明边界框
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                    
                    cv2.putText(frame, f"#{i+1} {label} {det.confidence:.2f}",
                               (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # 2. 绘制平滑后的检测（主要显示）
            if smoothed_result and USE_TEMPORAL_SMOOTHING:
                label = LABEL_MAP[smoothed_result['label']]
                smoothed_stats[label] += 1
                
                if smoothed_result['bbox'] is not None:
                    bbox = smoothed_result['bbox']
                    x1 = int(bbox[0] * frame.shape[1])
                    y1 = int(bbox[1] * frame.shape[0])
                    x2 = int(bbox[2] * frame.shape[1])
                    y2 = int(bbox[3] * frame.shape[0])
                    
                    # 颜色
                    if label == 'Green':
                        color = (0, 255, 0)
                    elif label == 'Red':
                        color = (0, 0, 255)
                    elif label == 'Yellow':
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 255)
                    
                    # 绘制粗边框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    
                    # 标签背景
                    label_text = f"{label} {smoothed_result['confidence']*100:.1f}%"
                    vote_text = f"({smoothed_result['votes']}/{smoothed_result['total_frames']})"
                    
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 35), (x1 + text_w + 20, y1), color, -1)
                    
                    # 标签文字
                    cv2.putText(frame, label_text, (x1 + 10, y1 - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(frame, vote_text, (x1 + 10, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # 稳定性指示器
                    stability = smoothed_result['votes'] / smoothed_result['total_frames']
                    stability_text = "STABLE" if stability >= 0.6 else "UNSTABLE"
                    stability_color = (0, 255, 0) if stability >= 0.6 else (0, 165, 255)
                    cv2.putText(frame, stability_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 2)
            
            elif best_detection and not USE_TEMPORAL_SMOOTHING:
                # 不使用平滑时，直接显示最佳检测
                label = LABEL_MAP[best_detection.label]
                detection_stats[label] += 1
                
                x1 = int(best_detection.xmin * frame.shape[1])
                x2 = int(best_detection.xmax * frame.shape[1])
                y1 = int(best_detection.ymin * frame.shape[0])
                y2 = int(best_detection.ymax * frame.shape[0])
                
                if label == 'Green':
                    color = (0, 255, 0)
                elif label == 'Red':
                    color = (0, 0, 255)
                elif label == 'Yellow':
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} {best_detection.confidence*100:.1f}%",
                           (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # === 显示状态信息 ===
            info_y = 25
            cv2.putText(frame, f"FPS: {fps:.1f}", (5, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            
            mode_text = "SMOOTH" if USE_TEMPORAL_SMOOTHING else "DIRECT"
            cv2.putText(frame, f"Mode: {mode_text}", (5, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 25
            
            # 阈值显示
            cv2.putText(frame, f"G:{CLASS_THRESHOLDS['Green']:.2f} "
                              f"R:{CLASS_THRESHOLDS['Red']:.2f} "
                              f"Y:{CLASS_THRESHOLDS['Yellow']:.2f}",
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Traffic Light Detection", frame)

            # === 键盘控制 ===
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('d'):
                DEBUG_MODE = not DEBUG_MODE
                print(f"\n[控制] 调试模式: {'开' if DEBUG_MODE else '关'}")
            elif key == ord('t'):
                USE_TEMPORAL_SMOOTHING = not USE_TEMPORAL_SMOOTHING
                print(f"\n[控制] 时序平滑: {'开' if USE_TEMPORAL_SMOOTHING else '关'}")
                if not USE_TEMPORAL_SMOOTHING:
                    smoother.reset()
            elif key == ord('c'):
                SHOW_ALL_CANDIDATES = not SHOW_ALL_CANDIDATES
                print(f"\n[控制] 显示候选: {'开' if SHOW_ALL_CANDIDATES else '关'}")
            elif key == ord('r'):
                smoother.reset()
                print("\n[控制] 平滑器已重置")
            elif key == ord('1'):
                CLASS_THRESHOLDS['Green'] = min(0.95, CLASS_THRESHOLDS['Green'] + 0.05)
                print(f"\n[控制] 绿灯阈值: {CLASS_THRESHOLDS['Green']:.2f}")
            elif key == ord('2'):
                CLASS_THRESHOLDS['Green'] = max(0.1, CLASS_THRESHOLDS['Green'] - 0.05)
                print(f"\n[控制] 绿灯阈值: {CLASS_THRESHOLDS['Green']:.2f}")
            elif key == ord('3'):
                CLASS_THRESHOLDS['Red'] = min(0.95, CLASS_THRESHOLDS['Red'] + 0.05)
                print(f"\n[控制] 红灯阈值: {CLASS_THRESHOLDS['Red']:.2f}")
            elif key == ord('4'):
                CLASS_THRESHOLDS['Red'] = max(0.1, CLASS_THRESHOLDS['Red'] - 0.05)
                print(f"\n[控制] 红灯阈值: {CLASS_THRESHOLDS['Red']:.2f}")
            elif key == ord('5'):
                CLASS_THRESHOLDS['Yellow'] = min(0.95, CLASS_THRESHOLDS['Yellow'] + 0.05)
                print(f"\n[控制] 黄灯阈值: {CLASS_THRESHOLDS['Yellow']:.2f}")
            elif key == ord('6'):
                CLASS_THRESHOLDS['Yellow'] = max(0.1, CLASS_THRESHOLDS['Yellow'] - 0.05)
                print(f"\n[控制] 黄灯阈值: {CLASS_THRESHOLDS['Yellow']:.2f}")

except Exception as e:
    print(f"\n❌ 程序异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        print("\n" + "=" * 70)
        print("运行统计")
        print("=" * 70)
        print(f"总帧数: {frame_count}")
        
        if USE_TEMPORAL_SMOOTHING:
            print(f"\n平滑后检测次数:")
            for label, count in smoothed_stats.items():
                print(f"  {label}: {count} 次")
        else:
            print(f"\n原始检测次数:")
            for label, count in detection_stats.items():
                print(f"  {label}: {count} 次")
        
        print("=" * 70)
    
    print("\n✓ 脚本结束。")