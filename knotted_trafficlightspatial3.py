import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
LABEL_MAP = ['red', 'green']
NN_INPUT_SIZE = 416

# --- 2. 构建最终的、最稳定的 DepthAI Pipeline ---
print("正在构建最终版Pipeline...")
pipeline = dai.Pipeline()

# --- 核心节点 ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

# --- 数据流节点 ---
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")
xout_spatial_data = pipeline.create(dai.node.XLinkOut)
xout_spatial_data.setStreamName("spatialData")

# --- 配置节点 ---
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumClasses(2)
detection_nn.setCoordinateSize(4)
detection_nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
detection_nn.setAnchorMasks({"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]})
detection_nn.setIouThreshold(0.5)
detection_nn.setBlobPath(MODEL_PATH)
detection_nn.input.setBlocking(False)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

# --- 链接节点 ---
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
cam_rgb.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
spatialLocationCalculator.out.link(xout_spatial_data.input)

print("Pipeline构建完成，正在连接设备...")

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_spatial = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
        configQueue = device.getInputQueue(name="spatialCalcConfig")
        
        frame = None
        detections = []

        while True:
            in_rgb = q_rgb.get()
            in_nn = q_nn.get()

            frame = in_rgb.getCvFrame()
            detections = in_nn.detections
            
            config = dai.SpatialLocationCalculatorConfig()
            if len(detections) > 0:
                for detection in detections:
                    # 使用裁剪后的坐标来创建ROI，确保有效性
                    xmin = max(0.0, detection.xmin)
                    ymin = max(0.0, detection.ymin)
                    xmax = min(1.0, detection.xmax)
                    ymax = min(1.0, detection.ymax)
                    if xmin >= xmax or ymin >= ymax: continue
                    
                    roi = dai.Rect(dai.Point2f(xmin, ymin), dai.Point2f(xmax, ymax))
                    cfg = dai.SpatialLocationCalculatorConfigData()
                    cfg.depthThresholds.lowerThreshold = 100
                    cfg.depthThresholds.upperThreshold = 10000
                    cfg.roi = roi
                    config.addROI(cfg)
            
            configQueue.send(config)
            
            in_spatial = q_spatial.get()
            spatialData = in_spatial.getSpatialLocations()

            if frame is not None:
                # 即使 spatialData 和 detections 长度不匹配，也尝试绘制
                # 以 detections 为主循环，安全地获取 spatialData
                for i, detection in enumerate(detections):
                    x1 = int(detection.xmin * frame.shape[1])
                    x2 = int(detection.xmax * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    y2 = int(detection.ymax * frame.shape[0])
                    
                    try: label = LABEL_MAP[detection.label]
                    except: label = "unknown"
                    
                    color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                    
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    
                    # 安全地获取空间数据
                    if i < len(spatialData):
                        coords = spatialData[i].spatialCoordinates
                        cv2.putText(frame, f"Z: {int(coords.z)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.imshow("Final Detector", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")