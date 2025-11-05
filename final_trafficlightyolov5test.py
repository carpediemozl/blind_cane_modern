import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/knottedblob/best10292_openvino_2022.1_6shave.blob'
LABEL_MAP = ['Green', 'Red', 'Yellow']  # ✅ 修正：与 data.yaml 一致
NN_INPUT_SIZE = 416

# 可调参数
CONF_THRESHOLD = 0.5  # 全局置信度阈值
SHOW_ONLY_BEST = True  # 只显示置信度最高的检测
DEBUG_MODE = True      # 显示调试信息

# 类别特定阈值（针对黄灯误检）
CLASS_THRESHOLDS = {
    'Green': 0.45,
    'Red': 0.45,
    'Yellow': 0.60  # 黄灯用更高阈值
}

# --- 2. 构建 Pipeline ---
print("=" * 60)
print("红绿灯检测系统 - 彩色框版本")
print("=" * 60)
print(f"模型: {MODEL_PATH}")
print(f"类别: {LABEL_MAP}")
print(f"全局阈值: {CONF_THRESHOLD}")
print(f"类别阈值: {CLASS_THRESHOLDS}")
print("=" * 60)

pipeline = dai.Pipeline()

# --- 定义节点 ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")

# --- 配置节点 ---
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
spatial_nn.setConfidenceThreshold(0.3)  # 较低的初始阈值，后续手动过滤
spatial_nn.input.setBlocking(False)
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)
spatial_nn.setDepthUpperThreshold(10000)

spatial_nn.setNumClasses(len(LABEL_MAP))
spatial_nn.setCoordinateSize(4)

# ✅ YOLOv5n 标准锚框（来自您的配置文件）
spatial_nn.setAnchors([
    10, 13, 16, 30, 33, 23,     # P3/8  (52x52)
    30, 61, 62, 45, 59, 119,    # P4/16 (26x26)
    116, 90, 156, 198, 373, 326 # P5/32 (13x13)
])
spatial_nn.setAnchorMasks({
    "side52": [0, 1, 2],  # P3/8
    "side26": [3, 4, 5],  # P4/16
    "side13": [6, 7, 8]   # P5/32
})
spatial_nn.setIouThreshold(0.5)

# --- 链接节点 ---
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
cam_rgb.preview.link(spatial_nn.input)
stereo.depth.link(spatial_nn.inputDepth)

spatial_nn.passthrough.link(xout_rgb.input)
spatial_nn.out.link(xout_nn.input)

print("\n正在连接设备...")

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("✓ 设备连接成功！")
        print(f"  DepthAI 版本: {dai.__version__}\n")
        print("按键说明:")
        print("  q - 退出")
        print("  d - 切换调试模式")
        print("  b - 切换最佳检测模式")
        print("  + - 提高全局阈值")
        print("  - - 降低全局阈值")
        print("  1 - 提高黄灯阈值")
        print("  2 - 降低黄灯阈值")
        print("=" * 60 + "\n")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        frame = None
        detections = []
        
        startTime = time.monotonic()
        counter = 0
        fps = 0
        frame_count = 0
        
        # 统计信息
        detection_stats = {label: 0 for label in LABEL_MAP}

        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()

            frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            frame_count += 1
            
            # FPS 计算
            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime
            
            # === 智能过滤检测结果 ===
            filtered_detections = []
            for det in detections:
                try:
                    label = LABEL_MAP[det.label]
                except:
                    label = "unknown"
                    continue
                
                # 使用类别特定阈值
                threshold = CLASS_THRESHOLDS.get(label, CONF_THRESHOLD)
                
                if det.confidence >= threshold:
                    filtered_detections.append(det)
            
            detections = filtered_detections
            
            # === 只保留最佳检测（可选）===
            if SHOW_ONLY_BEST and len(detections) > 0:
                best_detection = max(detections, key=lambda x: x.confidence)
                detections = [best_detection]
            
            # === 调试信息 ===
            if DEBUG_MODE and frame_count % 30 == 0:
                print(f"\n[帧 {frame_count}] FPS: {fps:.1f} | 检测数: {len(detections)}")
                if len(detections) == 0:
                    print("  无检测")
                else:
                    for i, det in enumerate(detections):
                        try:
                            label = LABEL_MAP[det.label]
                            coords = det.spatialCoordinates
                            print(f"  #{i+1} {label}: conf={det.confidence:.3f} | 距离={int(coords.z)}mm")
                        except:
                            pass

            # === 绘制检测结果（修复颜色逻辑）===
            for detection in detections:
                x1 = int(detection.xmin * frame.shape[1])
                x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                y2 = int(detection.ymax * frame.shape[0])
                
                try: 
                    label = LABEL_MAP[detection.label]
                    detection_stats[label] += 1
                except: 
                    label = "unknown"
                
                # === 修复：正确的颜色映射 (BGR 格式) ===
                if label == 'Green':
                    color = (0, 255, 0)      # 绿色
                elif label == 'Red':
                    color = (0, 0, 255)      # 红色
                elif label == 'Yellow':
                    color = (0, 255, 255)    # 黄色
                else:
                    color = (255, 255, 255)  # 白色（未知）
                
                coords = detection.spatialCoordinates
                
                # 绘制边界框（更粗）
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # 标签背景
                label_text = f"{label} {detection.confidence*100:.1f}%"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                
                # 标签文字（黑色）
                cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 距离信息
                distance_text = f"{int(coords.z/1000)}m"
                cv2.putText(frame, distance_text, (x1 + 5, y2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 详细坐标（可选，较小字体）
                cv2.putText(frame, f"X:{int(coords.x)} Y:{int(coords.y)} Z:{int(coords.z)}", 
                           (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # === 显示状态信息 ===
            info_y = 25
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            info_y += 30
            
            # 检测数
            cv2.putText(frame, f"Detections: {len(detections)}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            
            # 模式
            mode_text = "BEST" if SHOW_ONLY_BEST else "ALL"
            cv2.putText(frame, f"Mode: {mode_text}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 25
            
            # 阈值
            cv2.putText(frame, f"Global: {CONF_THRESHOLD:.2f}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(frame, f"Yellow: {CLASS_THRESHOLDS['Yellow']:.2f}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Traffic Light Detection", frame)

            # === 键盘控制 ===
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('d'):
                DEBUG_MODE = not DEBUG_MODE
                print(f"\n[控制] 调试模式: {'开启' if DEBUG_MODE else '关闭'}")
            elif key == ord('b'):
                SHOW_ONLY_BEST = not SHOW_ONLY_BEST
                mode = "只显示最佳" if SHOW_ONLY_BEST else "显示全部"
                print(f"\n[控制] 显示模式: {mode}")
            elif key == ord('+') or key == ord('='):
                CONF_THRESHOLD = min(0.95, CONF_THRESHOLD + 0.05)
                print(f"\n[控制] 全局阈值: {CONF_THRESHOLD:.2f}")
            elif key == ord('-') or key == ord('_'):
                CONF_THRESHOLD = max(0.1, CONF_THRESHOLD - 0.05)
                print(f"\n[控制] 全局阈值: {CONF_THRESHOLD:.2f}")
            elif key == ord('1'):
                CLASS_THRESHOLDS['Yellow'] = min(0.95, CLASS_THRESHOLDS['Yellow'] + 0.05)
                print(f"\n[控制] 黄灯阈值: {CLASS_THRESHOLDS['Yellow']:.2f}")
            elif key == ord('2'):
                CLASS_THRESHOLDS['Yellow'] = max(0.1, CLASS_THRESHOLDS['Yellow'] - 0.05)
                print(f"\n[控制] 黄灯阈值: {CLASS_THRESHOLDS['Yellow']:.2f}")

except Exception as e:
    print(f"\n❌ 程序异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    
    # 显示统计
    if frame_count > 0:
        print("\n" + "=" * 60)
        print("运行统计")
        print("=" * 60)
        print(f"总帧数: {frame_count}")
        print(f"\n各类别检测次数:")
        for label, count in detection_stats.items():
            print(f"  {label}: {count} 次")
        print("=" * 60)
    
    print("\n✓ 脚本结束。")