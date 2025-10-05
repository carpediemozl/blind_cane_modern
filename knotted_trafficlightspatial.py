import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/blob/trafficlight_frozen_darknet_yolov4_model_openvino_2021.3_5shave.blob'
LABEL_MAP = ['red', 'green']
NN_INPUT_SIZE = 416
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.1
NUM_CLASSES = 2
COORDINATE_SIZE = 4
ANCHORS = np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
ANCHOR_MASKS = {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}

# --- 2. 构建最终的 DepthAI Pipeline ---
print("正在构建Pipeline (主机端配置模式)...")
pipeline = dai.Pipeline()

# --- 定义节点 ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
xout_spatial_data = pipeline.create(dai.node.XLinkOut)
xout_spatial_data.setStreamName("spatialData")

# --- 配置节点 ---
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
detection_nn.setNumClasses(NUM_CLASSES)
detection_nn.setCoordinateSize(COORDINATE_SIZE)
detection_nn.setAnchors(ANCHORS)
detection_nn.setAnchorMasks(ANCHOR_MASKS)
detection_nn.setIouThreshold(IOU_THRESHOLD)
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
# 【核心修正】强制深度图的输出尺寸与RGB预览尺寸完全一致
#stereo.setOutputSize(cam_rgb.getPreviewSizeW(), cam_rgb.getPreviewSizeH())
stereo.setOutputSize(NN_INPUT_SIZE, NN_INPUT_SIZE)

# --- 链接节点 ---
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
cam_rgb.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_rgb.input)
detection_nn.out.link(xout_nn.input)
stereo.depth.link(xout_depth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
spatialLocationCalculator.out.link(xout_spatial_data.input)

print("Pipeline构建完成，正在连接设备...")

# --- 3. 主程序循环 ---
with dai.Device(pipeline) as device:
    # 获取输入和输出队列
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_spatial = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    # 【关键】获取用于发送配置的输入队列
    configQueue = device.getInputQueue(name="spatialCalcConfig")

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    fps = 0

    print("Final Spatial Traffic Light Detector is running...")
    print("Press 'q' to quit.")

    while True:
        # 使用 tryGet() 避免在循环开始时卡住
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        in_depth = q_depth.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        
        if in_depth is not None:
            depthFrame = in_depth.getFrame()
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
            cv2.imshow("depth", depthFrameColor)

        if in_nn is not None:
            detections = in_nn.detections
            counter += 1

        currentTime = time.monotonic()
        if (currentTime - startTime) > 1:
            fps = counter / (currentTime - startTime)
            counter = 0
            startTime = currentTime
        
        spatialData = [] # 默认空间数据为空

        # 【关键】主机端配置循环
        if len(detections) > 0:
            # 1. 创建一个新的配置对象
            config = dai.SpatialLocationCalculatorConfig()
            # 2. 遍历检测结果，为每个框创建ROI并添加到配置中
            for detection in detections:
                 # 【核心修正】在创建ROI之前，先裁剪坐标到 [0.0, 1.0] 的有效范围内
                xmin = max(0.0, detection.xmin)
                ymin = max(0.0, detection.ymin)
                xmax = min(1.0, detection.xmax)
                ymax = min(1.0, detection.ymax)
                #roi = dai.Rect(dai.Point2f(detection.xmin, detection.ymin), dai.Point2f(detection.xmax, detection.ymax))
                
                cfg = dai.SpatialLocationCalculatorConfigData()
                cfg.depthThresholds.lowerThreshold = 100
                cfg.depthThresholds.upperThreshold = 10000
                cfg.roi = roi
                config.addROI(cfg)
            # 3. 将这个包含所有ROI的配置发回给OAK-D
            configQueue.send(config)

            # 4. 获取对应的空间计算结果
            in_spatial = q_spatial.get() # 这里会阻塞，直到收到结果
            spatialData = in_spatial.getSpatialLocations()

        if frame is not None:
            # 将检测结果与空间数据进行匹配并绘制
            if len(detections) == len(spatialData):
                for i, detection in enumerate(detections):
                    x1 = int(detection.xmin * frame.shape[1])
                    x2 = int(detection.xmax * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0])
                    y2 = int(detection.ymax * frame.shape[0])
                    
                    try: label = LABEL_MAP[detection.label]
                    except: label = detection.label
                    
                    color = (0, 0, 255) if label == 'red' else (0, 255, 0)
                    coords = spatialData[i].spatialCoordinates
                    
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(coords.z)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"NN fps: {fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

print("程序已退出。")