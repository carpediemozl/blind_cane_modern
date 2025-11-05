import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/knottedblob/knottedtraffic1016.blob'
LABEL_MAP = ['green', 'red', 'yellow']  # 修正后的顺序
NN_INPUT_SIZE = 416

# 可调参数
CONF_THRESHOLD = 0.7  # 全局置信度阈值
SHOW_ONLY_BEST = True  # 是否只显示置信度最高的检测
DEBUG_MODE = True      # 是否显示调试信息
PRINT_INTERVAL = 30    # 每多少帧打印一次

# --- 2. 构建Pipeline ---
print("=" * 60)
print("交通灯检测系统 - 优化版")
print("=" * 60)
print(f"模型路径: {MODEL_PATH}")
print(f"类别映射: {LABEL_MAP}")
print(f"置信度阈值: {CONF_THRESHOLD}")
print(f"只显示最佳: {SHOW_ONLY_BEST}")
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

# === YOLO 配置 ===
spatial_nn.setBlobPath(MODEL_PATH)
spatial_nn.setConfidenceThreshold(0.3)  # 较低阈值，后续手动过滤
spatial_nn.input.setBlocking(False)
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)
spatial_nn.setDepthUpperThreshold(10000)

spatial_nn.setNumClasses(3)
spatial_nn.setCoordinateSize(4)
spatial_nn.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
spatial_nn.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
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
        print("  + - 提高置信度阈值")
        print("  - - 降低置信度阈值")
        print("  s - 显示统计信息")
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
        detection_history = []

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
            
            # === 过滤检测结果 ===
            filtered_detections = []
            for det in detections:
                if det.confidence >= CONF_THRESHOLD:
                    filtered_detections.append(det)
            
            detections = filtered_detections
            detection_history.append(len(detections))
            
            # === 只保留置信度最高的检测（可选）===
            if SHOW_ONLY_BEST and len(detections) > 0:
                # 按置信度排序，只保留最高的
                best_detection = max(detections, key=lambda x: x.confidence)
                detections = [best_detection]
            
            # === 调试信息打印 ===
            if DEBUG_MODE and frame_count % PRINT_INTERVAL == 0:
                print(f"\n[帧 {frame_count}] FPS: {fps:.1f} | 检测数: {len(detections)}")
                
                if len(detections) == 0:
                    print("  无检测")
                else:
                    for i, det in enumerate(detections):
                        try:
                            label = LABEL_MAP[det.label]
                        except:
                            label = f"unknown({det.label})"
                        
                        coords = det.spatialCoordinates
                        print(f"  #{i+1} {label}: "
                              f"conf={det.confidence:.3f} | "
                              f"距离={int(coords.z)}mm | "
                              f"位置=({int(coords.x)}, {int(coords.y)}, {int(coords.z)})")

            # === 绘制检测结果 ===
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
                
                # 颜色映射
                color_map = {
                    'red': (0, 0, 255),
                    'yellow': (0, 255, 255),
                    'green': (0, 255, 0)
                }
                color = color_map.get(label, (255, 255, 255))
                
                coords = detection.spatialCoordinates
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # 标签背景
                label_text = f"{label.upper()} {detection.confidence*100:.1f}%"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                
                # 标签文字
                cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # 距离信息
                distance_text = f"{int(coords.z/1000)}m"
                cv2.putText(frame, distance_text, (x1 + 5, y2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 详细空间坐标（较小字体）
                cv2.putText(frame, f"X:{int(coords.x)}", (x1 + 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, f"Y:{int(coords.y)}", (x1 + 5, y1 + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

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
            
            # 模式指示
            mode_text = "BEST" if SHOW_ONLY_BEST else "ALL"
            cv2.putText(frame, f"Mode: {mode_text}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 25
            
            # 阈值
            cv2.putText(frame, f"Thresh: {CONF_THRESHOLD:.2f}", 
                       (5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 平均检测数（右上角）
            if len(detection_history) > 30:
                avg_dets = sum(detection_history[-30:]) / 30
                cv2.putText(frame, f"Avg: {avg_dets:.1f}", 
                           (frame.shape[1] - 100, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
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
                print(f"\n[控制] 置信度阈值: {CONF_THRESHOLD:.2f}")
            elif key == ord('-') or key == ord('_'):
                CONF_THRESHOLD = max(0.1, CONF_THRESHOLD - 0.05)
                print(f"\n[控制] 置信度阈值: {CONF_THRESHOLD:.2f}")
            elif key == ord('s'):
                print(f"\n{'='*60}")
                print(f"统计信息 (总帧数: {frame_count})")
                print(f"{'='*60}")
                print(f"平均 FPS: {fps:.1f}")
                print(f"平均检测数: {sum(detection_history)/len(detection_history) if detection_history else 0:.2f}")
                print(f"\n各类别检测次数:")
                for label, count in detection_stats.items():
                    print(f"  {label}: {count} 次")
                print(f"{'='*60}\n")

except Exception as e:
    print(f"\n❌ 程序异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    
    # 最终统计
    if frame_count > 0:
        print("\n" + "=" * 60)
        print("运行统计")
        print("=" * 60)
        print(f"总帧数: {frame_count}")
        print(f"平均检测数: {sum(detection_history)/len(detection_history) if detection_history else 0:.2f}")
        print(f"\n各类别检测总计:")
        for label, count in detection_stats.items():
            print(f"  {label}: {count} 次")
        print("=" * 60)
    
    print("\n✓ 脚本结束。")