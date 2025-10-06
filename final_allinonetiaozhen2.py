import cv2
import depthai as dai
import numpy as np
import time

# --- 1. 核心参数 ---
MODEL_PATH = 'models/yolov8n_v6.blob'
LABEL_MAP = ['crosswalk', 'guide_arrows', 'blind_path', 'red_light', 'green_light']
NN_INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_CLASSES = 5
COORDINATE_SIZE = 4
ANCHORS = []
ANCHOR_MASKS = {}

# --- 2. 构建最终的 DepthAI Pipeline ---
print("正在构建'大一统'模型Pipeline...")
pipeline = dai.Pipeline()

# --- 定义节点 ---
cam_rgb = pipeline.create(dai.node.ColorCamera)
spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# 【回归】我们只需要两个输出：一个用于AI结果，一个用于高速预览
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("detections")
xout_preview = pipeline.create(dai.node.XLinkOut)
xout_preview.setStreamName("preview")

# --- 配置节点 ---
cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(30)

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
# 【关键优化】设置AI模型的输入队列大小和跳帧
spatial_nn.input.setQueueSize(1)
spatial_nn.input.setWaitForMessage(False)

# --- 链接节点 ---
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
cam_rgb.preview.link(spatial_nn.input)
stereo.depth.link(spatial_nn.inputDepth)
# 【回归】直接从相机获取高速预览流
cam_rgb.preview.link(xout_preview.input)
spatial_nn.out.link(xout_nn.input)

print("Pipeline构建完成，正在连接设备...")

# --- 3. 主程序循环 ---
try:
    with dai.Device(pipeline) as device:
        print("设备连接成功！")
        
        q_preview = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        
        frame = None
        latest_detections = []
        startTime = time.monotonic()
        ui_counter = 0
        nn_counter = 0
        ui_fps = 0
        nn_fps = 0

        COLORS = {
            "red_light": (0, 0, 255), "green_light": (0, 255, 0),
            "crosswalk": (255, 255, 0), "guide_arrows": (255, 255, 0),
            "blind_path": (255, 0, 255),
        }

        while True:
            # 【核心逻辑修正】
            # 1. 无条件获取最新的高速预览帧
            in_preview = q_preview.tryGet()
            if in_preview is not None:
                frame = in_preview.getCvFrame()
                ui_counter += 1

            # 2. “随缘”获取最新的AI结果，不等待
            in_nn = q_nn.tryGet()
            if in_nn is not None:
                latest_detections = in_nn.detections
                nn_counter += 1

            # 3. 每一帧都用最新的画面和最新的检测结果来绘图
            if frame is not None:
                for detection in latest_detections:
                    # ... (省略绘图代码) ...
                    x1 = int(detection.xmin * frame.shape[1]); x2 = int(detection.xmax * frame.shape[1])
                    y1 = int(detection.ymin * frame.shape[0]); y2 = int(detection.ymax * frame.shape[0])
                    try: label = LABEL_MAP[detection.label]
                    except: label = "unknown"
                    color = COLORS.get(label, (255, 255, 255))
                    coords = detection.spatialCoordinates
                    cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(frame, f"Z: {int(coords.z)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                currentTime = time.monotonic()
                if (currentTime - startTime) > 1:
                    ui_fps = ui_counter / (currentTime - startTime)
                    nn_fps = nn_counter / (currentTime - startTime)
                    ui_counter = 0
                    nn_counter = 0
                    startTime = currentTime
                
                cv2.putText(frame, f"UI_FPS: {ui_fps:.2f}", (2, frame.shape[0] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, f"NN_FPS: {nn_fps:.2f}", (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255))
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