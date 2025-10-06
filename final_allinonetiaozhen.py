import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 (从 V5allPipeline 中精确提取) ---
MODEL_PATH = 'models/yolov8n_v6.blob'
LABEL_MAP = ['crosswalk', 'guide_arrows', 'blind_path', 'red_light', 'green_light']
NN_INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.75 # 这是一个很高的值，如果检测不到物体，可以从这里开始降低
IOU_THRESHOLD = 0.1
NUM_CLASSES = 5
COORDINATE_SIZE = 4
ANCHORS = np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])
ANCHOR_MASKS = {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]}

# --- 2. 构建最终的 DepthAI Pipeline ---
print("正在构建'大一统'模型Pipeline...")
pipeline = dai.Pipeline()

# --- 定义节点 ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# 输出节点
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")
# 【新增】创建一个只输出原始预览画面的节点
xout_preview = pipeline.create(dai.node.XLinkOut)
xout_preview.setStreamName("preview")

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
spatial_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
spatial_nn.setNumClasses(NUM_CLASSES)
spatial_nn.setCoordinateSize(COORDINATE_SIZE)
spatial_nn.setAnchors(ANCHORS)
spatial_nn.setAnchorMasks(ANCHOR_MASKS)
spatial_nn.setIouThreshold(IOU_THRESHOLD)
spatial_nn.input.setBlocking(False)
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)
spatial_nn.setDepthUpperThreshold(10000)

# --- 链接节点 ---
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
# 【修改】将 cam_rgb.preview 同时链接到 NN输入 和 新的preview输出
cam_rgb.preview.link(spatial_nn.input)
cam_rgb.preview.link(xout_preview.input) # <--- 新增的链接
stereo.depth.link(spatial_nn.inputDepth)
spatial_nn.passthrough.link(xout_rgb.input)
spatial_nn.out.link(xout_nn.input)

print("Pipeline构建完成，正在连接设备...")

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        frame = None
        detections = []
        
        # 【新增】FPS计算器变量
        startTime = time.monotonic()
        counter = 0
        fps = 0

        # 为不同类别定义不同的颜色
        COLORS = {
            "red_light": (0, 0, 255), "green_light": (0, 255, 0),
            "crosswalk": (255, 255, 0), "guide_arrows": (255, 255, 0),
            "blind_path": (255, 0, 255),
        }

        # 【新增】跳帧处理的控制变量
        frame_skip = 6  # 每隔6帧进行一次AI推理 (30fps / 6 = 5fps)
        frame_count = 0
        latest_detections = [] # 用于缓存最新的检测结果
        latest_spatial_data = [] # 用于缓存最新的空间数据

        while True:
            # 【修改】每一帧都获取最新的摄像头画面
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            frame_count += 1
            # --- 条件性推理 ---
            # 只有在“推理帧”，我们才去获取AI的计算结果
            if frame_count >= frame_skip:
                frame_count = 0 # 重置计数器
            
                # 尝试获取最新的检测结果
                in_nn = q_nn.tryGet()
                if in_nn is not None:
                    # 【关键】将最新的结果缓存起来
                    latest_detections = in_nn.detections
        
            # 【修改】使用缓存的 detections 列表
            detections = latest_detections

            # 【新增】FPS计算逻辑
            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime

            for detection in detections:
                x1 = int(detection.xmin * frame.shape[1])
                x2 = int(detection.xmax * frame.shape[1])
                y1 = int(detection.ymin * frame.shape[0])
                y2 = int(detection.ymax * frame.shape[0])
                
                try: 
                    label = LABEL_MAP[detection.label]
                except: 
                    label = "unknown"
                
                color = COLORS.get(label, (255, 255, 255)) # 从字典获取颜色
                coords = detection.spatialCoordinates
                
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{detection.confidence * 100:.2f}%", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"Z: {int(coords.z)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 【新增】在主画面上显示FPS
            cv2.putText(frame, f"UI fps: {fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("Final All-in-One Detector", frame)

            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    print(f"\n程序在运行时出现异常: {e}")
    import traceback
    traceback.print_exc()

finally:
    cv2.destroyAllWindows()
    print("\n脚本执行结束。")